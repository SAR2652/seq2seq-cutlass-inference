import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def reshape_vec(vec):
    # vec shape: (2, batch_size, hidden_size)
    # Shape: (batch_size, 2, hidden_size)
    vec = vec.permute(1, 0, 2)
    # Shape: (batch_size, 1, 2 * hidden_size)
    vec = vec.reshape(vec.shape[0], 1, -1)
    # Shape: (1, batch_size, 2 * hidden_size)
    return vec.permute(1, 0, 2)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 bidirectional: bool = False, p: float = 0.2):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=bidirectional)

    def forward(self, inputs):
        embeddings = self.dropout(self.embedding(inputs))
        outputs, (hidden, cell) = self.lstm(embeddings)

        if self.bidirectional:      # Merge forward and backward hidden states
            hidden = reshape_vec(hidden)
            cell = reshape_vec(cell)

        return outputs, hidden, cell


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 use_cache: bool = False):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        msg = "Embedding dimension MUST be divisible by number of heads"
        assert embed_dim % num_heads == 0, msg

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys=None, value=None, use_cache: bool = False):

        batch_size, query_seq_len, _ = query.shape

        if keys is not None:
            _, keys_seq_len, _ = keys.shape
        else:
            keys = query
            keys_seq_len = query_seq_len

        if value is not None:
            _, value_seq_len, _ = keys.shape
        else:
            value = query
            value_seq_len = query_seq_len

        Q = self.query(query)

        # Original shape is (batch_size, seq_len, embed_dim)
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        # Permute dimensions to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.reshape(batch_size, query_seq_len, self.num_heads, self.head_dim)
        Q = Q.permute(0, 2, 1, 3)

        # Repeat for keys and values
        K = self.key(keys)
        V = self.value(value)
        K = K.reshape(batch_size, keys_seq_len, self.num_heads, self.head_dim)
        K = K.permute(0, 2, 1, 3)
        V = V.reshape(batch_size, value_seq_len, self.num_heads, self.head_dim)
        V = V.permute(0, 2, 1, 3)

        # Get attention scores of shape:
        # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attention_weights = self.softmax(attention_scores)

        # Attention output will have shape:
        # (batch_size, num_heads, seq_len, seq_len) *
        # (batch_size, num_heads, seq_len, head_dim) =
        # (batch_size, num_heads, seq_len, head_dim)
        attention_output = torch.matmul(attention_weights, V)
        # attention_output = torch.bmm(attention_weights, V)

        # Restore shape to (batch_size, seq_len, num_heads, head_dim)
        attention_output = attention_output.contiguous().permute(0, 2, 1, 3)

        # Reshape output to (batch_size, seq_len, embed_dim)
        attention_output = attention_output.reshape(batch_size, query_seq_len,
                                                    self.embed_dim)

        return attention_output


class MHADecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int,
                 encoder_bidirectional: bool, num_heads: int,
                 sos_token_id: int, max_length: int, device: torch.device,
                 p: float = 0.2):
        super(MHADecoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.encoder_bidirectional = encoder_bidirectional
        if encoder_bidirectional:
            self.hidden_dim *= 2
        self.num_heads = num_heads
        self.sos_token_id = sos_token_id
        self.max_length = max_length
        self.device = device
        self.embedding = nn.Embedding(vocab_size, self.hidden_dim)
        self.attention = MultiHeadAttention(self.hidden_dim, num_heads)
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(p)
        self.lstm = nn.LSTM(2 * self.hidden_dim, self.hidden_dim,
                            batch_first=True)
        self.fc_out = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, encoder_outputs, encoder_hidden, encoder_cell,
                targets=None):

        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long,
                                    device=self.device).fill_(
                                        self.sos_token_id)
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        decoder_outputs = list()

        for t in range(self.max_length):

            embedding = self.embedding(decoder_input)

            attention_output = self.attention(embedding, encoder_outputs,
                                              encoder_outputs)
            attention_output = self.layernorm(attention_output +
                                              self.dropout(attention_output))

            lstm_input = torch.cat((embedding, attention_output), dim=2)

            decoder_output, (decoder_hidden, decoder_cell) = \
                self.lstm(lstm_input, (decoder_hidden, decoder_cell))

            output = self.fc_out(decoder_output)

            decoder_outputs.append(output)

            if targets is not None:
                decoder_input = targets[:, t].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int,
                 attention_size: int):
        super(BahdanauAttention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_size = attention_size

        # Layers for calculating attention scores
        self.W = nn.Linear(encoder_hidden_size * 2, attention_size, bias=False)
        self.U = nn.Linear(decoder_hidden_size * 2, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Compute the Bahdanau attention scores and context vector.

        Args:
            decoder_hidden (Tensor): Current hidden state of the decoder
            (batch_size, decoder_hidden_size).
            encoder_outputs (Tensor): Outputs from the encoder
            (batch_size, seq_len, encoder_hidden_size).

        Returns:
            context_vector (Tensor): Weighted sum of encoder outputs
            (batch_size, encoder_hidden_size).
            attention_weights (Tensor): Attention weights
            (batch_size, seq_len).
        """
        # Expand decoder_hidden to match encoder_outputs dimensions
        # decoder_hidden = decoder_hidden.unsqueeze(1)
        # (batch_size, 1, decoder_hidden_size)

        _, seq_len, _ = encoder_outputs.shape
        # switch batch and seq_len dimensions for decoder hidden state
        decoder_hidden = decoder_hidden.transpose(0, 1)
        # (batch_size, 2, encoder_hidden_dim)
        # Merge forward and backward hidden states
        decoder_hidden = decoder_hidden.reshape(decoder_hidden.size(0), -1)

        decoder_hidden = decoder_hidden.unsqueeze(1)
        decoder_hidden = decoder_hidden.repeat(1, seq_len, 1)

        temp1 = self.W(encoder_outputs)
        # print(temp1.shape)
        temp2 = self.U(decoder_hidden)
        # print(temp2.shape)
        temp3 = torch.tanh(temp1 + temp2)

        # Calculate alignment scores
        score = self.v(temp3)
        # (batch_size, seq_len, 1)
        score = score.squeeze(-1)  # (batch_size, seq_len)

        # Compute attention weights
        attention_weights = F.softmax(score, dim=-1)  # (batch_size, seq_len)

        # Compute context vector as the weighted sum of encoder outputs
        context_vector = torch.bmm(attention_weights.unsqueeze(1),
                                   encoder_outputs)
        # (batch_size, 1, encoder_hidden_size)
        return context_vector, attention_weights


class DecoderWithBahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, embedding_dim: int,
                 encoder_hidden_dim: int, p: float = 0.1):
        super(DecoderWithBahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embedding_dim
        self.encoder_hidden_dim = encoder_hidden_dim

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Dropout Layer
        self.dropout = nn.Dropout(p)

        # Attention mechanism
        self.attention = BahdanauAttention(encoder_hidden_dim, hidden_dim,
                                           hidden_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim + encoder_hidden_dim * 2,
                            hidden_size=hidden_dim * 2, batch_first=True)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, input_token, hidden_state, cell_state, encoder_outputs):
        """
        Perform a decoding step.

        Args:
            input_token (Tensor): Input token indices (batch_size,).
            hidden_state (Tensor): Hidden state of the decoder
            (batch_size, hidden_dim).
            cell_state (Tensor): Cell state of the decoder
            (batch_size, hidden_dim).
            encoder_outputs (Tensor): Outputs from the encoder
            (batch_size, seq_len, encoder_hidden_dim).

        Returns:
            output (Tensor): Decoder output (batch_size, vocab_size).
            hidden_state (Tensor): Updated hidden state
            (batch_size, hidden_dim).
            cell_state (Tensor): Updated cell state
            (batch_size, hidden_dim).
            attention_weights (Tensor): Attention weights
            (batch_size, seq_len).
        """
        # Get embedding of input token
        input_embedding = self.dropout(
            self.embedding(input_token).unsqueeze(1)
            )

        # Compute attention context vector
        context_vector, attention_weights = self.attention(
            hidden_state, encoder_outputs
        )
        # print(context_vector.shape)
        # (batch_size, 1, encoder_hidden_dim)
        # print(input_embedding.shape)

        # Concatenate input embedding and context vector
        lstm_input = torch.cat((input_embedding, context_vector), dim=-1)
        # (batch_size, 1, embedding_dim + encoder_hidden_dim)

        # print(lstm_input.shape)
        # print(self.embed_dim + self.encoder_hidden_dim * 2)
        # print(hidden_state.shape)
        # print(cell_state.shape)

        hidden_state = hidden_state.permute(1, 0, 2)
        # (batch_size, num_directions, hidden_dim)
        hidden_state = hidden_state.reshape(hidden_state.size(0), -1) \
            .unsqueeze(0)

        cell_state = cell_state.permute(1, 0, 2)
        # (batch_size, num_directions, hidden_dim)
        cell_state = cell_state.reshape(cell_state.size(0), -1) \
            .unsqueeze(0)

        # Pass through LSTM
        lstm_output, (hidden_state, cell_state) = self.lstm(
            lstm_input, (hidden_state, cell_state)
            )

        # print(lstm_output.shape)
        # Compute output
        output = self.fc(lstm_output.squeeze(1))  # (batch_size, vocab_size)
        # print(output.shape)
        return output, hidden_state, cell_state, attention_weights


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,
                 encoder_hidden_dim: int, decoder_hidden_dim: int,
                 sos_token_id: int, target_len: int, device: torch.device):
        super(Seq2SeqModel, self).__init__()

        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, embed_dim, encoder_hidden_dim)
        self.decoder = DecoderWithBahdanauAttention(
            decoder_hidden_dim, vocab_size, embed_dim, encoder_hidden_dim
            )
        self.sos_token_id = sos_token_id
        self.target_len = target_len
        self.device = device

    def forward(self, inputs, teacher_force_ratio: float = 0.5, targets=None,
                eval: bool = False):

        encoder_outputs, decoder_hidden_state, decoder_cell_state = \
            self.encoder(inputs)

        batch_size, _, _ = encoder_outputs.shape
        use_teacher_forcing = random.random() < teacher_force_ratio
        use_teacher_forcing = torch.tensor([use_teacher_forcing] *
                                           batch_size).to(self.device)

        decoder_input = torch.tensor([self.sos_token_id] * batch_size,
                                     dtype=torch.int32).to(self.device)
        # print(decoder_input)

        outputs = torch.zeros(batch_size, self.target_len,
                              self.vocab_size).to(self.device)
        # best_guesses = np.array

        if eval:
            best_guesses = np.ones((batch_size, self.target_len))

        for t in range(1, self.target_len):

            logits, decoder_hidden_state, decoder_cell_state, _ = \
                self.decoder(decoder_input, decoder_hidden_state,
                             decoder_cell_state, encoder_outputs)

            # print(logits.shape)
            best_guess = logits.argmax(-1)
            outputs[:, t, :] = logits

            if not eval:
                # decoder_input = torch.where(
                #     # Align dimensions for broadcasting
                #     use_teacher_forcing.unsqueeze(1),
                #     # Use target token (teacher forcing)
                #     targets[:, t].unsqueeze(1),
                #     best_guess.unsqueeze(1)     # Use model's predicted token
                # ).squeeze(1)
                decoder_input = targets[:, t]
                # print(decoder_input)
            else:
                decoder_input = best_guess
                best_guess_np = best_guess.detach().cpu().numpy()
                best_guesses[:, t] = best_guess_np
                # print(best_guesses)
                # best_guesses.append(best_guess.item())

        # print(outputs.shape)
        if not eval:
            return outputs
        else:
            return outputs, best_guesses


class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int,
                 embed_dim: int, num_heads: int):

        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim,
                            batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, target_token, encoder_outputs, decoder_hidden_state,
                decoder_cell_state):

        # Shape = (B, 1, E)
        embedded = self.embedding(target_token)

        # Shape = (B, 1, H)
        context = self.attention(embedded, encoder_outputs, encoder_outputs)

        # Shape = (B, 1, E + H)
        lstm_input = torch.cat((embedded, context), dim=2)

        outputs, (hidden, cell) = self.lstm(lstm_input,
                                            (decoder_hidden_state,
                                             decoder_cell_state))

        # Shape = (B, 1, V)
        predictions = self.fc_out(outputs).squeeze(1)
        return predictions, hidden, cell


class Seq2SeqModelSA(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 encoder_hidden_dim: int, decoder_hidden_dim: int,
                 sos_token_id: int, target_len: int, device: torch.device):
        super(Seq2SeqModelSA, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(vocab_size, embed_dim, encoder_hidden_dim)
        self.decoder = Decoder(decoder_hidden_dim, vocab_size, embed_dim * 2,
                               num_heads)
        self.sos_token_id = sos_token_id
        self.target_len = target_len
        self.device = device

    def forward(self, inputs, teacher_force_ratio: float = 0.5, targets=None,
                eval: bool = True):

        encoder_outputs, decoder_hidden_state, decoder_cell_state = \
            self.encoder(inputs)

        # print(encoder_outputs.shape)
        # print(decoder_hidden_state.shape)
        # print(decoder_cell_state.shape)

        batch_size, _, _ = encoder_outputs.shape
        use_teacher_forcing = random.random() < teacher_force_ratio
        use_teacher_forcing = torch.tensor([use_teacher_forcing] *
                                           batch_size).to(self.device)

        decoder_input = torch.tensor([self.sos_token_id] * batch_size,
                                     dtype=torch.int32).to(self.device)
        # print(decoder_input.shape)

        outputs = torch.zeros(batch_size, self.target_len,
                              self.vocab_size).to(self.device)
        # print(outputs.shape)

        for t in range(1, self.target_len):

            # logits, decoder_hidden_state, decoder_cell_state, _ = \
            logits, decoder_hidden_state, decoder_cell_state = \
                self.decoder(decoder_input, encoder_outputs,
                             decoder_hidden_state, decoder_cell_state,)

            outputs[:, t, :] = logits

            if targets is not None:
                decoder_input = targets[:, t].unsqueeze(1)
                print(decoder_input.shape)
            # else:
            #     decoder_input

        return outputs


class DecoderSACA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, vocab_size: int):
        super(DecoderSACA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Self-Attention for decoder's own tokens (causal)
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)

        # Cross-Attention (queries from decoder, keys/values from encoder)
        self.cross_attention = MultiHeadAttention(embed_dim, num_heads)

        self.lstm = nn.LSTM(2 * embed_dim, embed_dim,
                            batch_first=True)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, target_token: torch.Tensor,
                encoder_outputs: torch.Tensor,
                decoder_hidden_state: torch.Tensor,
                decoder_cell_state: torch.Tensor, eval: bool = False,
                kv_cache: torch.Tensor = None,
                decoder_tokens: torch.Tensor = None,
                decoder_step: int = 0):
        """
        Args:
            target_token: (B, 1) Last generated token.
            encoder_outputs: (B, S, H) Encoder outputs (Keys & Values for
            cross-attention).
            decoder_hidden_state: (1, B, H) Previous LSTM hidden state.
            decoder_cell_state: (1, B, H) Previous LSTM cell state.

        Returns:
            next_token_logits: (B, V) Predicted token logits for next step.
            hidden: (1, B, H) Updated hidden state.
            cell: (1, B, H) Updated cell state.
        """

        # 1. Embed token (B, 1, E)
        embedded = self.embedding(target_token)

        if kv_cache is not None:
            kv_cache[:, decoder_step, :] = embedded.squeeze(1)

        # 2. Apply Self-Attention (no mask needed for one token)
        if decoder_step > 0:
            if kv_cache is None:
                context_embeds = self.embedding(decoder_tokens)
            else:
                context_embeds = kv_cache[:, :decoder_step, :]

            self_attn_output = self.self_attention(embedded, context_embeds,
                                                   context_embeds)
        else:
            self_attn_output = self.self_attention(embedded, embedded,
                                                   embedded)

        # print(f'Self Attention Output Shape = {self_attn_output.shape}')

        # 3. Apply Cross-Attention (Queries from Self-Attention Output, Keys &
        # Values from Encoder)
        cross_attn_output = self.cross_attention(self_attn_output,
                                                 encoder_outputs,
                                                 encoder_outputs)

        # print(f'Cross Attention Output Shape = {cross_attn_output.shape}')
        # print(f'Embedded Shape = {embedded.shape}')

        # 4. LSTM processes the combined representation
        lstm_input = torch.cat((embedded, cross_attn_output),
                               dim=2)  # (B, 1, E + H)
        outputs, (hidden, cell) = self.lstm(lstm_input, (decoder_hidden_state,
                                                         decoder_cell_state))

        # 5. Predict next token
        next_token_logits = self.fc_out(outputs)  # (B, 1, V)

        return next_token_logits.squeeze(1), hidden, cell


class CrossAttentionModel(nn.Module):
    def __init__(self, enc_embed_dim: int, hidden_dim: int, vocab_size: int,
                 num_heads: int, sos_token_id: int,
                 device: torch.device, bidirectional: bool = False,
                 teacher_force_ratio: float = 0.5, use_cache: bool = False):
        super(CrossAttentionModel, self).__init__()
        self.enc_embed_dim = enc_embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.sos_token_id = sos_token_id
        self.bidirectional = bidirectional
        self.teacher_force_ratio = teacher_force_ratio
        self.encoder = Encoder(vocab_size, enc_embed_dim, hidden_dim,
                               bidirectional)

        if bidirectional:
            hidden_dim *= 2

        self.decoder = DecoderSACA(hidden_dim, num_heads, vocab_size)
        self.device = device
        self.use_cache = use_cache

    def forward(self, inputs, targets=None):

        encoder_outputs, decoder_hidden_state, decoder_cell_state = \
            self.encoder(inputs)

        # print(f'Encoder Outputs Shape = {encoder_outputs.shape}')
        # print(f'Decoder Hidden Shape = {decoder_hidden_state.shape}')
        # print(f'Decoder Cell Shape = {decoder_cell_state.shape}')

        batch_size, target_len, _ = encoder_outputs.shape

        outputs = torch.zeros(batch_size, target_len,
                              self.vocab_size).to(self.device)

        # KV cache
        context_tokens = None
        kv_cache = None
        if self.use_cache:
            kv_cache = torch.zeros(batch_size, target_len, self.hidden_dim)
        else:
            context_tokens = torch.empty(batch_size, 0)

        decoder_input = torch.tensor([self.sos_token_id] * batch_size) \
            .unsqueeze(1).to(self.device)

        for t in range(target_len):

            logits, decoder_hidden_state, decoder_cell_state = self.decoder(
                decoder_input, encoder_outputs, decoder_hidden_state,
                decoder_cell_state, kv_cache, context_tokens, t
            )

            outputs[:, t, :] = logits

            use_teacher_forcing = random.random() < self.teacher_force_ratio

            if targets is not None and use_teacher_forcing:
                decoder_input = targets[:, t].unsqueeze(1)
                # print(f'Training Shape = {decoder_input.shape}')
            else:
                # print(f'Eval Logits Shape = {logits.shape}')
                decoder_input = logits.argmax(-1)

                # print(f'Eval Shape 1 = {decoder_input.shape}')
                decoder_input = decoder_input.unsqueeze(1)
                # print(f'Eval Shape 2 = {decoder_input.shape}')

        return outputs
