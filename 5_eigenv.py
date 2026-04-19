import numpy as np

# ---------------------------------------------------------
# Método de Jacobi
# ---------------------------------------------------------
def jacobi(A, b, x0=None, tolerancia=1e-6, max_iteracoes=100):
    """
    Resolve Ax = b usando o Método Iterativo de Jacobi.
    Atualiza todas as variáveis em simultâneo no final de cada iteração.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    # Se não for dada uma estimativa inicial, começa com zeros
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    x_novo = np.zeros(n)
    
    for k in range(max_iteracoes):
        for i in range(n):
            # Soma de A[i,j] * x[j] para todos os j diferentes de i
            soma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            
            # Isola x_i na equação i
            x_novo[i] = (b[i] - soma) / A[i, i]
            
        # Critério de paragem: a distância máxima entre o x novo e o x antigo
        # Usamos a norma do máximo (np.linalg.norm com ord=np.inf)
        erro = np.linalg.norm(x_novo - x, ord=np.inf) / np.linalg.norm(x_novo, ord=np.inf)
        if erro < tolerancia:
            return x_novo
            
        # O x passa a ser o x_novo para a próxima iteração
        # Usamos .copy() para não ligar as referências de memória
        x = x_novo.copy()
        
    raise ValueError("O Método de Jacobi não convergiu.")

# ---------------------------------------------------------
# Método de Gauss-Seidel
# ---------------------------------------------------------
def gauss_seidel(A, b, x0=None, tolerancia=1e-6, max_iteracoes=100):
    """
    Resolve Ax = b usando o Método Iterativo de Gauss-Seidel.
    Utiliza o valor mais recente das variáveis mal elas são calculadas.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    
    for k in range(max_iteracoes):
        x_antigo = x.copy()
        
        for i in range(n):
            # No Gauss-Seidel, se já calculámos o x[0] nesta iteração, 
            # já usamos o novo x[0] para calcular o x[1], e assim por diante!
            soma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - soma) / A[i, i]
            
        erro = np.linalg.norm(x - x_antigo, ord=np.inf) / np.linalg.norm(x, ord=np.inf)
        if erro < tolerancia:
            return x
            
    raise ValueError("O Método de Gauss-Seidel não convergiu.")

# ---------------------------------------------------------
# Método SOR (Relaxação Sucessiva)
# ---------------------------------------------------------
def sor(A, b, omega, x0=None, tolerancia=1e-6, max_iteracoes=100):
    """
    Resolve Ax = b usando o Método SOR (Successive Over-Relaxation).
    omega: Fator de relaxação (0 < omega < 2). 
           Se omega = 1, o método é idêntico ao Gauss-Seidel.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    
    for k in range(max_iteracoes):
        x_antigo = x.copy()
        
        for i in range(n):
            soma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            
            # 1. Calcula o passo "Gauss-Seidel" puro
            x_gs = (b[i] - soma) / A[i, i]
            
            # 2. Faz uma média ponderada entre o valor antigo e o valor Gauss-Seidel
            x[i] = (1 - omega) * x_antigo[i] + omega * x_gs
            
        erro = np.linalg.norm(x - x_antigo, ord=np.inf) / np.linalg.norm(x, ord=np.inf)
        if erro < tolerancia:
            return x
            
    raise ValueError("O Método SOR não convergiu.")

# Exemplo de utilização global (Matriz estritamente diagonal dominante):
# A = [[10, 2, 1], 
#      [1, 5, 1], 
#      [2, 3, 10]]
# b = [7, -8, 6]
# print("Jacobi:", jacobi(A, b))
# print("Gauss-Seidel:", gauss_seidel(A, b))
# print("SOR (omega=1.2):", sor(A, b, omega=1.2))