import math
import numpy as np 
from scipy.interpolate import approximate_taylor_polynomial

# Método Babilônico

# É uma aplicação do método de Newton-Raphson usada especificamente 
# para calcular raízes quadradas através de aproximações sucessivas

def metodo_babilonico(S, estimativa_inicial=None, tolerancia=1e-10, max_iteracoes=100):
    """
    Calcula a raiz quadrada de S usando o Método Babilônico.
    """
    if S < 0:
        raise ValueError("Não é possível calcular a raiz de um número negativo.")
    if S == 0:
        return 0.0

    # Se não houver estimativa, começamos com o próprio S ou S/2
    x = estimativa_inicial if estimativa_inicial else S / 2.0
    
    for i in range(max_iteracoes):
        # Fórmula de iteração: x_{n+1} = 0.5 * (x_n + S / x_n)
        proximo_x = 0.5 * (x + S / x)
        
        # Critério de parada: diferença absoluta entre iterações
        if abs(proximo_x - x) < tolerancia:
            return proximo_x
            
        x = proximo_x
        
    return x

# Exemplo de uso:
# print(metodo_babilonico(25))


# Séries de Taylor (com Multiplicações Aninhadas)

# Usadas para calcular funções como e x. 
# Para evitar o cálculo pesado de potências complexas e garantir 
# estabilidade numérica, o polinômio da série é reescrito 
# colocando o x em evidência sucessivamente

def exp_taylor_aninhada(x, n=20):
    """
    Calcula e^x usando a Série de Taylor com multiplicações aninhadas (Método de Horner).
    n: número de termos (grau do polinômio).
    """
    resultado = 1.0
    # Iteramos de n até 1 para construir a estrutura aninhada de dentro para fora
    for i in range(n, 0, -1):
        resultado = 1 + (x / i) * resultado
    return resultado

# Exemplo de uso:
# print(exp_taylor_aninhada(1.0)) # Próximo a 2.71828


def exp_scipy_taylor(x, grau=10):
    # Aproxima a função np.exp em torno do ponto 0 (Série de Maclaurin)
    polinomio = approximate_taylor_polynomial(np.exp, 0, grau, scale=1)
    return polinomio(x)

# Para uso em produção, apenas use:
# np.exp(x)


# Aproximação Numérica de Derivadas

# Quando a derivada analítica é difícil de obter, usa-se a diferença entre pontos 
# próximos (x e x+h). Pode ser Progressiva (olha para frente),
# Regressiva (olha para trás) ou Central (olha para os dois lados,]
# possuindo o menor erro de truncamento matemático, mas exigindo mais avaliações da função)

def derivada_numerica(f, x, h=1e-5, tipo='central'):
    """
    Calcula a derivada de f em x usando diferenças finitas.
    tipo: 'progressiva', 'regressiva' ou 'central'
    """
    if tipo == 'progressiva':
        # f'(x) ≈ (f(x + h) - f(x)) / h
        return (f(x + h) - f(x)) / h
    
    elif tipo == 'regressiva':
        # f'(x) ≈ (f(x) - f(x - h)) / h
        return (f(x) - f(x - h)) / h
    
    elif tipo == 'central':
        # f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h)
        return (f(x + h) - f(x - h)) / (2 * h)
    
    else:
        raise ValueError("Tipo desconhecido. Use 'progressiva', 'regressiva' ou 'central'.")

# Exemplo de uso:
# print(derivada_numerica(math.sin, math.pi, tipo='central')) # Deve resultar em ~ -1.0
