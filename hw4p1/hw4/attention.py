import torch

class Softmax:

    '''
    DO NOT MODIFY! AN INSTANCE IS ALREADY SET IN THE Attention CLASS' CONSTRUCTOR. USE IT!
    Performs softmax along the last dimension
    '''
    def forward(self, Z):

        z_original_shape = Z.shape

        self.N = Z.shape[0]*Z.shape[1]
        self.C = Z.shape[2]
        Z = Z.reshape(self.N, self.C)

        Ones_C = torch.ones((self.C, 1))
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)

        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):

        dLdA_original_shape = dLdA.shape

        dLdA = dLdA.reshape(self.N, self.C)

        dLdZ = torch.zeros((self.N, self.C))
        
        for i in range(self.N):

            J = torch.zeros((self.C, self.C))

            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        J[m, n] = -self.A[i][m] * self.A[i][n]

            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ.reshape(dLdA_original_shape)

class Attention:
        
        def __init__(self, weights_keys, weights_queries, weights_values):

            """
            Initialize instance variables. Refer to writeup for notation.
            input_dim = D, key_dim = query_dim = D_k, value_dim = D_v

            Argument(s)
            -----------
            
            weights_keys (torch.tensor, dim = (D X D_k)): weight matrix for keys 
            weights_queries (torch.tensor, dim = (D X D_k)): weight matrix for queries 
            weights_values (torch.tensor, dim = (D X D_v)): weight matrix for values 
            
            """

            # Store the given weights as parameters of the class.
            self.W_k    = weights_keys # TODO
            self.W_q    = weights_queries # TODO
            self.W_v    = weights_values # TODO

            # Use this object to perform softmax related operations.
            # It performs softmax over the last dimension which is what you'll need.
            self.softmax = Softmax()
            
        def forward(self, X):

            """
            Compute outputs of the self-attention layer.
            Stores keys, queries, values, raw and normalized attention weights.
            Refer to writeup for notation.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
            You should permute only the required axes.

            Input
            -----
            X (torch.tensor, dim = (B, T, D)): Input batch

            Return
            ------
            X_new (torch.tensor, dim = (B, T, D_v)): Output batch

            """

            self.X = X
        
            # Compute the values of Key, Query and Value

            self.Q = X @ self.W_q # TODO
            self.K = X @ self.W_k # TODO
            self.V = X @ self.W_v # TODO

            # Calculate unormalized Attention Scores (logits)

            self.A_w    =  (self.Q @ self.K.transpose(-2,-1)) / (self.K.shape[-1] ** 0.5) # TODO

            # Create additive causal attention mask and apply mask
            # Hint: Look into torch.tril/torch.triu and account for batch dimension

            attn_mask    = torch.tril(torch.ones(self.A_w.shape[-2:], device = X.device)).unsqueeze(0) # TODO
            self.A_w = self.A_w.masked_fill(attn_mask == 0, float('-inf'))

            # Calculate/normalize Attention Scores

            self.A_sig   = self.softmax.forward(self.A_w) # TODO

            # Calculate Attention context 

            X_new         = self.A_sig @ self.V # TODO

            return X_new
            
        def backward(self, dLdXnew):
            """
            Backpropagate derivatives through the self-attention layer.
            Stores derivatives wrt keys, queries, values, and weight matrices.
            batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

            Input
            -----
            dLdXnew (torch.tensor, dim = (B, T, D_v)): Derivative of the loss wrt attention layer outputs

            Return
            ------
            dLdX (torch.tensor, dim = (B, T, D)): Derivative of the loss wrt attention layer inputs
            """

            # Compute the scaling factor (sqrt(D_k))
            scale = self.K.shape[-1] ** 0.5

            # Derivatives wrt attention weights (raw and normalized)
            dLdA_sig = torch.matmul(dLdXnew, self.V.transpose(-2, -1))  # Shape: (B, T, T)
            
            # Incorporate the scaling factor
            dLdA_w = self.softmax.backward(dLdA_sig) / scale        # Shape: (B, T, T)

            # Derivatives wrt keys, queries, and values
            self.dLdV = torch.matmul(self.A_sig.transpose(-2, -1), dLdXnew)  # Shape: (B, T, D_v)
            self.dLdK = torch.matmul(dLdA_w.transpose(-2, -1), self.Q)       # Shape: (B, T, D_k)
            self.dLdQ = torch.bmm(dLdA_w, self.K)                           # Shape: (B, T, D_k)

            # Derivatives wrt weight matrices using einsum
            self.dLdWq = torch.einsum("btd,btk->dk", self.X, self.dLdQ)    # Shape: (D, D_k)
            self.dLdWv = torch.einsum("btd,btv->dv", self.X, self.dLdV)    # Shape: (D, D_v)
            self.dLdWk = torch.einsum("btd,btk->dk", self.X, self.dLdK)    # Shape: (D, D_k)

            # Derivative wrt input
            dLdX = (
                torch.matmul(self.dLdQ, self.W_q.transpose(0, 1)) +
                torch.matmul(self.dLdK, self.W_k.transpose(0, 1)) +
                torch.matmul(self.dLdV, self.W_v.transpose(0, 1))
            )  # Shape: (B, T, D)

            return dLdX
