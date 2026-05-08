import numpy as np

# ---------------------------------------------------------
# Método da Potência
# ---------------------------------------------------------
def metodo_potencia(A, v0=None, tolerancia=1e-6, max_iteracoes=1000):
    """
    Encontra o autovalor DOMINANTE (maior em módulo) e o seu autovetor.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    # Chute inicial (vetor aleatório se não for fornecido)
    v = np.random.rand(n) if v0 is None else np.array(v0, dtype=float)
    v = v / np.linalg.norm(v) # Normalização inicial
    
    lambda_antigo = 0.0
    
    for i in range(max_iteracoes):
        # 1. Multiplica o vetor pela matriz
        y = A @ v
        
        # 2. Estima o autovalor atual (Quociente de Rayleigh)
        lambda_atual = np.dot(v, y)
        
        # 3. Normaliza o novo vetor para evitar que cresça para o infinito
        v_novo = y / np.linalg.norm(y)
        
        # 4. Critério de paragem (autovalor estabilizou)
        if abs(lambda_atual - lambda_antigo) < tolerancia:
            return lambda_atual, v_novo
            
        v = v_novo
        lambda_antigo = lambda_atual
        
    raise ValueError("O Método da Potência não convergiu.")

# ---------------------------------------------------------
# Método da Iteração Inversa
# ---------------------------------------------------------
def iteracao_inversa(A, v0=None, tolerancia=1e-6, max_iteracoes=1000):
    """
    Encontra o autovalor de MENOR módulo e o seu autovetor.
    Equivale a aplicar o Método da Potência na matriz A^-1, 
    mas resolvendo o sistema Ay = v em vez de inverter a matriz.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    v = np.random.rand(n) if v0 is None else np.array(v0, dtype=float)
    v = v / np.linalg.norm(v)
    
    lambda_antigo = 0.0
    
    for i in range(max_iteracoes):
        # Em vez de y = A @ v, resolvemos o sistema A @ y = v
        # (Na prática, deveríamos usar fatoração LU (ou Eliminação Gaussiana) apenas uma vez, 
        # mas aqui usamos o solver do numpy para simplificar)
        y = np.linalg.solve(A, v)
        
        # Normaliza
        v_novo = y / np.linalg.norm(y)
        
        # Quociente de Rayleigh com o vetor original
        lambda_atual = np.dot(v_novo, A @ v_novo)
        
        if abs(lambda_atual - lambda_antigo) < tolerancia:
            return lambda_atual, v_novo
            
        v = v_novo
        lambda_antigo = lambda_atual
        
    raise ValueError("O Método da Iteração Inversa não convergiu.")

# ---------------------------------------------------------
# Método da Iteração Inversa com Quociente de Rayleigh
# ---------------------------------------------------------
def iteracao_inversa_rayleigh(A, mi0, v0=None, tolerancia=1e-6, max_iteracoes=1000):
    """
    Encontra o autovalor MAIS PRÓXIMO do valor 'mi0' e o seu autovetor.
    Convergência cúbica fantástica!
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    I = np.eye(n) # Matriz Identidade
    
    v = np.random.rand(n) if v0 is None else np.array(v0, dtype=float)
    v = v / np.linalg.norm(v)
    
    mi = mi0 # 'mi' (mu) é a nossa estimativa atual do autovalor
    
    for i in range(max_iteracoes):
        # Deslocamos a matriz A usando a estimativa atual 'mi'
        A_deslocada = A - mi * I
        
        # Resolvemos o sistema (A - mi*I)y = v
        try:
            y = np.linalg.solve(A_deslocada, v)
        except np.linalg.LinAlgError:
            # Se a matriz for singular, mi é um autovalor exato!
            return mi, v
            
        v_novo = y / np.linalg.norm(y)
        
        # Atualizamos a estimativa mi com o novo Quociente de Rayleigh
        mi_novo = np.dot(v_novo, A @ v_novo)
        
        if abs(mi_novo - mi) < tolerancia:
            return mi_novo, v_novo
            
        v = v_novo
        mi = mi_novo
        
    raise ValueError("A Iteração com Quociente de Rayleigh não convergiu.")

# Exemplo de utilização (Matriz Simétrica):
# A = [[4, 1], [1, 3]]
# print("Dominante:", metodo_potencia(A)[0])
# print("Menor:", iteracao_inversa(A)[0])
# print("Próximo de 3:", iteracao_inversa_rayleigh(A, mi0=3.2)[0])


import scipy.linalg as la

def autovalores_scipy(A):
    # Retorna um array de autovalores e uma matriz onde as colunas são os autovetores
    autovalores, autovetores = la.eig(A)
    # Pega apenas a parte real
    return np.real(autovalores)