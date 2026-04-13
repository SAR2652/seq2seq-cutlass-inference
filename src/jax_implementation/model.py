import jax
import jax.numpy as jnp

from enum import Enum
import flax.linen as nn


class EncoderFLAX(nn.Module):
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    bidirectional: bool = False

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_dim)
        self.forward_lstm = nn.LSTMCell(self.hidden_dim)
        if self.bidirectional:
            self.backward_lstm = nn.LSTMCell(self.hidden_dim)

    def __call__(self, inputs):
        """ Forward pass of encoder """
        embeddings = self.embedding(inputs)
        # print(embeddings.shape)
        batch_size, seq_len, _ = embeddings.shape

        fwd_hidden = jnp.zeros((batch_size, self.hidden_dim))
        fwd_cell = jnp.zeros((batch_size, self.hidden_dim))

        # print('hidden and cell states organized')

        outputs = []
        # Iterate over sequence
        for t in range(seq_len):
            (fwd_hidden, fwd_cell), _ = self.forward_lstm(
                (fwd_hidden, fwd_cell), embeddings[:, t, :])
            outputs.append(fwd_hidden)

        outputs = jnp.stack(outputs, axis=1)
        # Convert list to array

        # print('outputs organized')

        if self.bidirectional:
            bkwd_hidden = jnp.zeros((batch_size, self.hidden_dim))
            bkwd_cell = jnp.zeros((batch_size, self.hidden_dim))
            backward_outputs = []
            # Iterate over sequence
            for t in range(seq_len - 1, -1, -1):
                (bkwd_hidden, bkwd_cell), out = self.backward_lstm(
                    (bkwd_hidden, bkwd_cell), embeddings[:, t, :])
                backward_outputs.append(bkwd_hidden)

            backward_outputs = backward_outputs[::-1]  # align t=0..seq_len-1
            backward_outputs = jnp.stack(backward_outputs, axis=1)
            outputs = jnp.concatenate([outputs, backward_outputs], axis=-1)

            hidden = jnp.concatenate([fwd_hidden, bkwd_hidden], axis=-1)
            cell = jnp.concatenate([fwd_cell, bkwd_cell], axis=-1)

        else:
            hidden = fwd_hidden
            cell = fwd_cell

        # print(f'Encoder Outputs Shape = {outputs.shape}')
        # print(f'Encoder Hidden Shape = {hidden.shape}')
        # print(f'Encoder Cell Shape = {cell.shape}')

        return outputs, hidden, cell


class Mode(Enum):
    NONE = "none"
    SELF = "self"
    CROSS = "cross"


class MultiHeadAttentionFLAX(nn.Module):
    embed_dim: int
    num_heads: int
    mode: Enum

    def setup(self):
        assert self.embed_dim % self.num_heads == 0, \
            "Embedding dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        # Linear transformations for Q, K, V
        self.query_proj = nn.Dense(self.embed_dim)
        self.key_proj = nn.Dense(self.embed_dim)
        self.value_proj = nn.Dense(self.embed_dim)

        # Output projection
        self.out_proj = nn.Dense(self.embed_dim)

    def __call__(self, query: jnp.ndarray, key: jnp.ndarray = None,
                 value: jnp.ndarray = None, decoder_step: int = None,
                 kv_cache: jnp.ndarray = None):
        batch_size, query_seq_len, _ = query.shape

        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = query

        _, key_seq_len, _ = key.shape
        _, value_seq_len, _ = value.shape

        # Project queries, keys, and values
        Q = self.query_proj(query)  # (batch_size, query_seq_len, embed_dim)

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        Q = Q.reshape(batch_size, query_seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))

        # Repeat the above operations for Keys & Values
        # if:
        # 1. Attention mode is "cross" and kv_cache is not None but
        # kv_cache['key'] is None and kv_cache['value'] is None.
        # Cross Attention is done between encoder outputs and current token.
        # Encoder outputs are constant, hece calculate K/V only once and
        # cache them for later use
        # 2. Attention mode is "self"
        # Self Attention occurs between current token and all previous tokens

        if self.mode in [Mode.SELF, Mode.NONE] or \
            (self.mode == Mode.CROSS and isinstance(kv_cache, dict) and
             kv_cache['key'] is None and kv_cache['value'] is None):

            # Calculate K/V matrices
            K = self.key_proj(key)      # (batch_size, key_seq_len, embed_dim)
            V = self.value_proj(value)
            # (batch_size, value_seq_len, embed_dim)
            K = K.reshape(batch_size, key_seq_len, self.num_heads,
                          self.head_dim)
            V = V.reshape(batch_size, value_seq_len, self.num_heads,
                          self.head_dim)
            K = jnp.transpose(K, (0, 2, 1, 3))
            V = jnp.transpose(V, (0, 2, 1, 3))

            # Do this only if Attention Mode is "cross"
            # Assignment Step for Cross Attention K/V values IF using KV Cache
            if self.mode == Mode.CROSS:
                kv_cache['key'] = K
                kv_cache['value'] = V

        if self.mode in [Mode.SELF, Mode.CROSS] and kv_cache is not None:

            # Do this only if Attention Mode is "self"
            # Assign new token's K,V vectors for Self Attention only
            if self.mode == Mode.SELF:
                kv_cache['key'] = \
                    kv_cache['key'] \
                    .at[:, :, decoder_step: decoder_step + 1, :].set(K)
                kv_cache['value'] = \
                    kv_cache['value'] \
                    .at[:, :, decoder_step: decoder_step + 1, :].set(V)

            K = kv_cache['key']
            V = kv_cache['value']

        # Compute scaled dot-product attention
        attention_scores = jnp.einsum("bhqd, bhkd -> bhqk", Q, K) * self.scale

        # (batch_size, num_heads, query_seq_len, key_seq_len)
        attention_weights = nn.softmax(attention_scores.astype(jnp.float32),
                                       axis=-1)

        # Compute attention output
        attention_output = jnp.einsum("bhqk, bhkd -> bhqd", attention_weights,
                                      V)

        # Restore shape: (batch_size, query_seq_len, num_heads, head_dim)
        attention_output = jnp.transpose(attention_output, (0, 2, 1, 3))

        # Reshape back to (batch_size, query_seq_len, embed_dim)
        attention_output = attention_output.reshape(batch_size, query_seq_len,
                                                    self.embed_dim)

        if kv_cache is not None:
            return self.out_proj(attention_output), kv_cache

        return self.out_proj(attention_output), None


class DecoderSACAFLAX(nn.Module):
    embed_dim: int
    num_heads: int
    vocab_size: int
    use_cache: bool

    def setup(self):
        # Token embedding layer
        self.embedding = nn.Embed(self.vocab_size, self.embed_dim)

        if self.use_cache:
            self_attn_mode = Mode.SELF
            cross_attn_mode = Mode.CROSS
        else:
            self_attn_mode = Mode.NONE
            cross_attn_mode = Mode.NONE

        # Self-Attention (decoder attending to its own past tokens)
        self.self_attention = MultiHeadAttentionFLAX(self.embed_dim,
                                                     self.num_heads,
                                                     self_attn_mode)

        # Cross-Attention (queries from decoder, keys/values from encoder)
        self.cross_attention = MultiHeadAttentionFLAX(self.embed_dim,
                                                      self.num_heads,
                                                      cross_attn_mode)

        # LSTM processes combined attention representations
        self.lstm = nn.LSTMCell(self.embed_dim)

        # Final projection to vocabulary size
        self.fc_out = nn.Dense(self.vocab_size)

    def __call__(self, target_token: jnp.ndarray, encoder_outputs: jnp.ndarray,
                 hidden_state: jnp.ndarray, cell_state: jnp.ndarray,
                 decoder_tokens: jnp.ndarray = None, decoder_step: int = 0,
                 self_attn_kv_cache: jnp.ndarray = None,
                 cross_attn_kv_cache: jnp.ndarray = None):
        """
        Args:
            target_token: (B, 1) Last generated token.
            encoder_outputs: (B, S, H) Encoder outputs (keys & values for
            cross-attention).
            hidden_state: (B, H) Previous LSTM hidden state.
            cell_state: (B, H) Previous LSTM cell state.

        Returns:
            next_token_logits: (B, 1, V) Predicted token logits for next step.
            new_hidden: (B, H) Updated hidden state.
            new_cell: (B, H) Updated cell state.
        """

        # 1. Embed the target token
        embedded = self.embedding(target_token)  # (B, 1, E)

        # print(f'Embedded = {embedded.shape}')

        # 2. Apply Self-Attention
        if decoder_step > 0 and self_attn_kv_cache is None:
            context_embeds = self.embedding(
                decoder_tokens[:, :decoder_step]
            )
            self_attn_output, self_attn_kv_cache = self.self_attention(
                embedded, context_embeds, context_embeds, decoder_step,
                self_attn_kv_cache
            )
        else:
            self_attn_output, self_attn_kv_cache = \
                self.self_attention(embedded, embedded, embedded,
                                    decoder_step, self_attn_kv_cache)

        # print(f'Self Attn = {self_attn_output.shape}')

        # 3. Apply Cross-Attention (queries from Self-Attention Output,
        # keys/values from encoder)
        # (B, 1, E)
        cross_attn_output, cross_attn_kv_cache = \
            self.cross_attention(self_attn_output, encoder_outputs,
                                 encoder_outputs, kv_cache=cross_attn_kv_cache)

        # print(f'Cross Attn = {cross_attn_output.shape}')

        # 4. Concatenate embeddings and attention output
        lstm_input = jnp.concatenate([embedded, cross_attn_output],
                                     axis=-1)  # (B, 1, 2E)

        # lstm_ip_hidden = jnp.zeros((batch_size, self.embed_dim * 2))
        # lstm_ip_cell = jnp.zeros((batch_size, self.embed_dim * 2))
        # print(f'LSTM Input = {lstm_input.shape}')
        # print(f'Hidden Shape = {hidden_state.shape}')
        # print(f'Cell Shape = {cell_state.shape}')

        # 5. LSTM step (JAX LSTMCell operates per timestep)
        (hidden_state, cell_state), _ = self.lstm(
            (hidden_state, cell_state), lstm_input[:, 0, :]
        )

        # print(f'Out Hidden Shape = {hidden_state.shape}')
        # print(f'Out Shape = {out.shape}')

        # 6. Predict next token
        next_token_logits = self.fc_out(hidden_state)  # (B, 1, V)

        return next_token_logits, hidden_state, cell_state


class CrossAttentionModelFLAX(nn.Module):
    enc_embed_dim: int
    hidden_dim: int
    vocab_size: int
    num_heads: int
    sos_token_id: int
    bidirectional: bool = False
    use_cache: bool = False
    teacher_force_ratio: float = 0.5

    def setup(self):
        self.encoder = EncoderFLAX(self.vocab_size, self.enc_embed_dim,
                                   self.hidden_dim, self.bidirectional)

        hidden_dim = self.hidden_dim * 2 if self.bidirectional else \
            self.hidden_dim
        self.decoder = DecoderSACAFLAX(hidden_dim, self.num_heads,
                                       self.vocab_size, self.use_cache)
        self.updated_hidden_dim = hidden_dim

    def __call__(self, inputs: jnp.ndarray, targets: jnp.ndarray = None,
                 eval: bool = False, curr_epoch: int = None,
                 warmup_epochs: int = None):
        """
        Args:
            inputs: (B, S) Input token indices.
            targets: (B, T) Target token indices (optional, for teacher
            forcing).
            rng: PRNG key for random number generation.

        Returns:
            outputs: (B, T, V) Logits for each token position.
        """

        # Encode input sequence
        encoder_outputs, decoder_hidden_state, decoder_cell_state = \
            self.encoder(inputs)

        batch_size = encoder_outputs.shape[0]
        target_len = targets.shape[1] if targets is not None \
            else encoder_outputs.shape[1]
        outputs = jnp.zeros((batch_size, target_len, self.vocab_size))

        if not self.use_cache:
            context_tokens = jnp.zeros((batch_size, target_len),
                                       dtype=jnp.int32)
            self_attn_kv_cache = None
            cross_attn_kv_cache = None
        else:
            context_tokens = None
            self_attn_kv_cache = {
                'key': jnp.zeros((batch_size, self.num_heads, target_len,
                                 self.updated_hidden_dim // self.num_heads)),
                'value': jnp.zeros((batch_size, self.num_heads, target_len,
                                   self.updated_hidden_dim // self.num_heads))
            }
            cross_attn_kv_cache = {
                'key': None,
                'value': None
            }

        # Initial decoder input: <SOS> token
        decoder_input = jnp.full((batch_size, 1), self.sos_token_id)

        for t in range(target_len):

            logits, decoder_hidden_state, decoder_cell_state = self.decoder(
                decoder_input, encoder_outputs, decoder_hidden_state,
                decoder_cell_state, context_tokens, t, self_attn_kv_cache,
                cross_attn_kv_cache
            )

            # print(f'Logits Shape = {logits.shape}')

            outputs = outputs.at[:, t, :].set(logits)

            # Teacher forcing during training at all times for a fixed number
            # of epochs
            if not eval and curr_epoch is not None and \
                    warmup_epochs is not None and curr_epoch < warmup_epochs:
                if targets is not None:
                    decoder_input = targets[:, t:t+1]  # Use ground truth
                else:
                    raise ValueError("Targets required during training with "
                                     "teacher forcing.")

            else:
                probs = jax.nn.softmax(logits, axis=-1)
                decoder_input = jnp.argmax(probs, axis=-1, keepdims=True)

                if not self.use_cache:
                    context_tokens = context_tokens.at[:, t:t + 1].set(
                        decoder_input.reshape((batch_size, 1))
                    )

        return outputs
