import numpy as np

# ---------------------------------------------------------
# Retro-substituição
# ---------------------------------------------------------
def retro_substituicao(U, y):
    """
    Resolve o sistema triangular superior Ux = y.
    U: Matriz triangular superior (n x n)
    y: Vetor de termos independentes (n)
    """
    n = len(y)
    x = np.zeros(n)
    
    # Começamos na última linha e vamos subindo (de n-1 até 0)
    for i in range(n - 1, -1, -1):
        # A soma dos produtos das variáveis já conhecidas pelos seus coeficientes
        soma = sum(U[i, j] * x[j] for j in range(i + 1, n))
        
        # Isola a variável x[i]
        x[i] = (y[i] - soma) / U[i, i]
        
    return x

# ---------------------------------------------------------
# Eliminação Gaussiana (com e sem Pivotamento)
# ---------------------------------------------------------
def eliminacao_gaussiana(A, b, pivotamento=True):
    """
    Resolve Ax = b usando Eliminação Gaussiana.
    pivotamento: Se True, usa pivotamento parcial para estabilidade numérica.
    """
    # Converte para float para evitar erros de divisão com inteiros
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Cria a matriz aumentada [A | b] para fazer as operações numa só estrutura
    # np.c_ concatena arrays lado a lado (por colunas)
    M = np.c_[A, b]
    
    # O ciclo de eliminação (anda pela diagonal principal)
    for k in range(n - 1):
        
        # 1. Pivotamento Parcial (Troca de Linhas)
        if pivotamento:
            # Encontra o índice do maior elemento em módulo na coluna k (da linha k para baixo)
            max_idx = k + np.argmax(abs(M[k:, k]))
            
            # Se o maior elemento for 0, o sistema não tem solução única (matriz singular)
            if M[max_idx, k] == 0:
                raise ValueError("A matriz é singular (não tem solução única).")
                
            # Troca a linha atual (k) com a linha do maior elemento
            # (Isto reduz erros de arredondamento e evita divisão por zero)
            M[[k, max_idx]] = M[[max_idx, k]]
            
        else:
            # Sem pivotamento, se o pivô atual for zero, o método falha
            if M[k, k] == 0:
                raise ValueError("Pivô nulo encontrado. Ative o pivotamento para continuar.")

        # 2. Eliminação (Zerar elementos debaixo do pivô)
        for i in range(k + 1, n):
            # O fator multiplicador: elemento_a_zerar / pivo
            fator = M[i, k] / M[k, k]
            
            # Atualiza toda a linha i subtraindo a linha do pivô multiplicada pelo fator
            M[i, :] = M[i, :] - fator * M[k, :]
            
    # Após o ciclo, M é uma matriz triangular superior
    # Extraímos a nova matriz U e o novo vetor y
    U = M[:, :n]
    y = M[:, n]
    
    # Chamamos a retro-substituição para encontrar a solução final
    return retro_substituicao(U, y)

# Exemplo de utilização:
# A = [[2, 1, -1],
#      [-3, -1, 2],
#      [-2, 1, 2]]
# b = [8, -11, -3]
# x = eliminacao_gaussiana(A, b, pivotamento=True)
# print("A solução é:", x)


import numpy as np

def resolver_sistema_numpy(A, b):
    # Encontra a solução para Ax = b no milissegundo!
    return np.linalg.solve(A, b)

