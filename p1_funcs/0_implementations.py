import importlib
import math
import numpy as np

# ---------------------------------------------------------
# IMPORTAÇÃO DOS MÓDULOS (Lidando com nomes iniciados por números)
# ---------------------------------------------------------
mod1 = importlib.import_module("1_functions")
mod2 = importlib.import_module("2_roots")
mod3 = importlib.import_module("3_polynomials")
mod4 = importlib.import_module("4_equations")
mod5 = importlib.import_module("5_eigenv")
mod6 = importlib.import_module("6_stationary")
mod7 = importlib.import_module("7_non_stationary")

def executar_testes():
    print("="*50)
    print("   TESTES DE CÁLCULO NUMÉRICO")
    print("="*50)

    # ---------------------------------------------------------
    print("\n[1] FUNÇÕES (Aproximações e Derivadas)")
    print("-" * 30)
    S = 25
    print(f"Método Babilônico (Raiz de {S}):", mod1.metodo_babilonico(S))
    
    x_taylor = 1.0
    print(f"Série de Taylor (e^{x_taylor}):", mod1.exp_taylor_aninhada(x_taylor))
    
    f_sin = math.sin
    x_derivada = math.pi
    print(f"Derivada Numérica de sin(pi):", mod1.derivada_numerica(f_sin, x_derivada, tipo='central'))

    # ---------------------------------------------------------
    print("\n[2] ZEROS DE FUNÇÕES (Raízes)")
    print("-" * 30)
    # f(x) = x^2 - 4  --> Raiz exata é 2 (e -2)
    f_raiz = lambda x: x**2 - 4
    df_raiz = lambda x: 2*x
    d2f_raiz = lambda x: 2
    
    print("Bissecção [0, 5]:", mod2.bisseccao(f_raiz, 0, 5))
    print("Posição Falsa [0, 5]:", mod2.posicao_falsa(f_raiz, 0, 5))
    print("Newton-Raphson (x0=5):", mod2.newton_raphson(f_raiz, df_raiz, x0=5))
    print("Secante (x0=4, x1=5):", mod2.secante(f_raiz, x0=4, x1=5))
    print("Halley (x0=5):", mod2.halley(f_raiz, df_raiz, d2f_raiz, x0=5))
    print("Brent [0, 5]:", mod2.brent(f_raiz, a=0, b=5))
    
    # f_muller(x) = x^3 - 2x - 5 (Raiz real ~2.0945)
    f_muller = lambda x: x**3 - 2*x - 5
    print("Muller (x0=1, x1=1.5, x2=3):", mod2.muller(f_muller, 1.0, 1.5, 3.0))

    # ---------------------------------------------------------
    print("\n[3] POLINÓMIOS")
    print("-" * 30)
    # P(x) = x^3 + 2x^2 - 5x - 6 (Raízes: -3, -1, 2)
    coefs = [1, 2, -5, -6]
    print("Regras de Inspeção:", mod3.regras_inspecao(coefs))
    print("Cotas de Kojima/Cauchy/Fujiwara:", mod3.cotas_raizes(coefs))
    print("Newton-Horner (x0=3):", mod3.newton_horner(coefs, x0=3.0))
    
    # P(x) = x^4 - 2x^3 + 4x^2 - 4x + 4 (Raízes complexas)
    coefs_bair = [1, -2, 4, -4, 4]
    print("Bairstow (Todas as raízes):", mod3.bairstow(coefs_bair))

    # ---------------------------------------------------------
    print("\n[4] SISTEMAS DE EQUAÇÕES (Métodos Diretos)")
    print("-" * 30)
    A_eq = [[2, 1, -1],
            [-3, -1, 2],
            [-2, 1, 2]]
    b_eq = [8, -11, -3]
    x_eq = mod4.eliminacao_gaussiana(A_eq, b_eq, pivotamento=True)
    print("Eliminação Gaussiana com Pivotamento:\n x =", x_eq)

    # ---------------------------------------------------------
    print("\n[5] SISTEMAS DE EQUAÇÕES (Métodos Iterativos)")
    print("-" * 30)
    # Matriz estritamente diagonal dominante
    A_iter = [[10, 2, 1], 
              [1, 5, 1], 
              [2, 3, 10]]
    b_iter = [7, -8, 6]
    print("Jacobi:", mod5.jacobi(A_iter, b_iter))
    print("Gauss-Seidel:", mod5.gauss_seidel(A_iter, b_iter))
    print("SOR (omega=1.2):", mod5.sor(A_iter, b_iter, omega=1.2))

    # ---------------------------------------------------------
    print("\n[6] OTIMIZAÇÃO E MÉTODOS ESTACIONÁRIOS")
    print("-" * 30)
    # Matriz simétrica definida positiva
    A_stat = [[4, 1], 
              [1, 3]]
    b_stat = [1, 2]
    print("Método do Gradiente (Cauchy):", mod6.metodo_gradiente(A_stat, b_stat))
    print("Gradientes Conjugados (CG):", mod6.gradientes_conjugados(A_stat, b_stat))

    # ---------------------------------------------------------
    print("\n[7] AUTOVALORES E AUTOVETORES (Não Estacionários)")
    print("-" * 30)
    A_eig = [[4, 1], 
             [1, 3]]
    # Devolvem um tuplo (autovalor, autovetor). Apanhamos o índice [0] para ver só o valor
    print("Método da Potência (Autovalor Dominante):", mod7.metodo_potencia(A_eig)[0])
    print("Iteração Inversa (Menor Autovalor):", mod7.iteracao_inversa(A_eig)[0])
    print("Rayleigh (Mais próximo de 3.2):", mod7.iteracao_inversa_rayleigh(A_eig, mi0=3.2)[0])

    print("\n" + "="*50)

# Ponto de entrada do script
if __name__ == "__main__":
    executar_testes()