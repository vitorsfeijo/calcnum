import numpy as np

# ---------------------------------------------------------
# Método do Gradiente (G) ou Cauchy (Steepest Descent)
# ---------------------------------------------------------
def metodo_gradiente(A, b, x0=None, tolerancia=1e-6, max_iteracoes=1000):
    """
    Resolve Ax = b deslizando sempre na direção da descida mais íngreme (o gradiente).
    A matriz A tem de ser simétrica e definida positiva.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    for i in range(max_iteracoes):
        # O resíduo r é a direção de descida (r = b - Ax)
        r = b - A @ x
        
        # Critério de paragem
        if np.linalg.norm(r) < tolerancia:
            return x
            
        # O quão longe devemos andar nesta direção? (Cálculo do alpha ótimo)
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        
        # Dá o passo!
        x = x + alpha * r
        
    raise ValueError("O Método do Gradiente não convergiu.")

# ---------------------------------------------------------
# Método das Direções Conjugadas (CD)
# (Conceito Teórico: Usar direções A-ortogonais. O CG abaixo é a sua melhor versão)
# ---------------------------------------------------------

# ---------------------------------------------------------
# Método dos Gradientes Conjugados (CG)
# ---------------------------------------------------------
def gradientes_conjugados(A, b, x0=None, tolerancia=1e-6, max_iteracoes=1000):
    """
    Resolve Ax = b usando direções de busca inteligentes ("conjugadas").
    Garante chegar à solução exata em, no máximo, 'n' passos matemáticos!
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    # O primeiro passo é igual ao de Cauchy
    r = b - A @ x
    p = r.copy() # 'p' é a direção de busca

    for i in range(max_iteracoes):
        if np.linalg.norm(r) < tolerancia:
            return x
            
        Ap = A @ p
        r_dot_r = np.dot(r, r)
        
        # Tamanho do passo na direção p
        alpha = r_dot_r / np.dot(p, Ap)
        
        # Atualiza a posição e recalcula o resíduo
        x = x + alpha * p
        r_novo = r - alpha * Ap
        
        # Fator mágico (beta) que corrige a próxima curva
        # Impede que andemos para trás ou repitamos direções passadas
        beta = np.dot(r_novo, r_novo) / r_dot_r
        
        # A nova direção é uma mistura da descida atual com a inércia passada
        p = r_novo + beta * p
        
        r = r_novo
        
    return x

# Exemplo de utilização (Matriz Simétrica Definida Positiva):
# A = [[4, 1], 
#      [1, 3]]
# b = [1, 2]
# print("Cauchy:", metodo_gradiente(A, b))
# print("Gradiente Conjugado:", gradientes_conjugados(A, b))

from scipy.sparse import linalg

def resolver_cg_scipy(A, b):
    # Devolve a solução e um código indicando sucesso
    x, info = linalg.cg(A, b)
    return x

