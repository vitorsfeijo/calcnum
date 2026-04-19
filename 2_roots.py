from scipy import optimize
import cmath 

# Iteração do Ponto-Fixo

# No arquivo 2_roots.py

def iteracao_ponto_fixo(g, x0, tolerancia=1e-10, max_iteracoes=100):
    """
    Encontra a raiz de f(x) = 0 resolvendo o problema equivalente x = g(x).
    
    Parâmetros:
    g : A função de iteração x = g(x)
    x0: Estimativa inicial
    tolerancia: Critério de parada (diferença entre iterações)
    max_iteracoes: Proteção contra loops infinitos
    """
    x = x0
    
    for i in range(max_iteracoes):
        proximo_x = g(x)
        
        # Critério de parada: a diferença entre o novo x e o anterior é pequena o suficiente?
        if abs(proximo_x - x) < tolerancia:
            return proximo_x
            
        x = proximo_x
        
    # Se o loop terminar sem retornar, o método divergiu ou precisava de mais iterações
    raise ValueError(f"O método não convergiu após {max_iteracoes} iterações.")

# Exemplo de uso:
# Para encontrar a raiz de f(x) = x^2 - x - 2 = 0
# Podemos reescrever como x = sqrt(x + 2). Assim, g(x) = sqrt(x + 2)
# import math
# g = lambda x: math.sqrt(x + 2)
# print(iteracao_ponto_fixo(g, x0=1.0)) # Deve convergir para 2.0


def iteracao_scipy(g, x0=1.0):
    return optimize.fixed_point(g, x0)

# Passamos a função g e o chute inicial x0
# iteracao_scipy(g, 1.0)



# Métodos de Enquadramento (Seguros, mas lentos):
# Bissecção

def bisseccao(f, a, b, tolerancia=1e-6, max_iteracoes=100):
    """
    Encontra a raiz de f(x) no intervalo [a, b] usando o Método da Bissecção.
    """
    # Verificação inicial do Teorema de Bolzano
    if f(a) * f(b) >= 0:
        raise ValueError("A função deve ter sinais opostos em a e b (f(a) * f(b) < 0).")
    
    iteracao = 0
    while iteracao < max_iteracoes:
        # Calcula o ponto médio
        c = (a + b) / 2.0
        
        # Critério de paragem: se a raiz foi encontrada exatamenta ou o intervalo é menor que a tolerância
        if f(c) == 0 or (b - a) / 2.0 < tolerancia:
            return c
            
        # Decide qual metade do intervalo manter
        # Se f(a) e f(c) têm sinais opostos, a raiz está entre a e c
        if f(a) * f(c) < 0:
            b = c
        # Caso contrário, a raiz está entre c e b
        else:
            a = c
            
        iteracao += 1
        
    raise ValueError("O método não convergiu no número máximo de iterações.")

# Exemplo de utilização:
# Encontrar a raiz de f(x) = x^2 - 4 no intervalo [0, 5]
# f = lambda x: x**2 - 4
# raiz = bisseccao(f, 0, 5)
# print(f"A raiz é aproximadamente: {raiz}")

def bisseccao_scipy(f, a, b):
    # A SciPy trata os detalhes e tem uma tolerância (xtol) padrão muito precisa
    return optimize.bisect(f, a, b)


# Posição Falsa

def posicao_falsa(f, a, b, tolerancia=1e-6, max_iteracoes=100):
    """
    Encontra a raiz de f(x) no intervalo [a, b] usando o Método da Posição Falsa.
    """
    # Verificação do Teorema de Bolzano
    if f(a) * f(b) >= 0:
        raise ValueError("A função deve ter sinais opostos em a e b (f(a) * f(b) < 0).")
    
    for i in range(max_iteracoes):
        # Fórmula da Posição Falsa (interseção da secante com o eixo x)
        # c = a - f(a) * (b - a) / (f(b) - f(a)) # (Forma alternativa)
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        
        # Critério de paragem: o valor da função em c é suficientemente pequeno?
        # Também podemos usar |b - a| < tolerancia, mas na Posição Falsa 
        # um dos limites costuma ficar "preso", pelo que f(c) ≈ 0 é mais seguro.
        if abs(f(c)) < tolerancia:
            return c
            
        # Decide qual lado do intervalo substituir
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
            
    raise ValueError("O método não convergiu no número máximo de iterações.")

# Exemplo de utilização:
# f = lambda x: x**2 - 4
# print(posicao_falsa(f, 0, 5))




# Métodos de Busca ou Abertos (Rápidos, mas exigem boas estimativas iniciais):
# Newton-Raphson

def newton_raphson(f, df, x0, tolerancia=1e-6, max_iteracoes=100):
    """
    Encontra a raiz de f(x) usando o Método de Newton-Raphson.
    
    Parâmetros:
    f : Função cuja raiz queremos encontrar
    df: A derivada analítica da função f
    x0: Estimativa inicial
    """
    x = x0
    
    for i in range(max_iteracoes):
        fx = f(x)
        dfx = df(x)
        
        # Proteção contra divisão por zero (se a tangente for horizontal, o método falha)
        if dfx == 0:
            raise ValueError(f"A derivada zerou em x = {x}. O método falhou.")
            
        # Fórmula de Newton-Raphson
        proximo_x = x - fx / dfx
        
        # Critério de paragem
        if abs(proximo_x - x) < tolerancia:
            return proximo_x
            
        x = proximo_x
        
    raise ValueError("O método não convergiu no número máximo de iterações.")

# Exemplo de utilização:
# f = lambda x: x**2 - 4
# df = lambda x: 2*x  # Derivada de x^2 - 4
# print(newton_raphson(f, df, x0=5))


def newton_scipy(f, df, x0):
    # Passamos a função, o chute inicial e a derivada (fprime)
    return optimize.newton(f, x0=x0, fprime=df)


# Secante

def secante(f, x0, x1, tolerancia=1e-6, max_iteracoes=100):
    """
    Encontra a raiz de f(x) usando o Método da Secante.
    
    Parâmetros:
    f : Função cuja raiz queremos encontrar
    x0: Primeira estimativa inicial
    x1: Segunda estimativa inicial
    """
    for i in range(max_iteracoes):
        fx0 = f(x0)
        fx1 = f(x1)
        
        # Proteção contra divisão por zero (se a linha secante for horizontal)
        if fx1 - fx0 == 0:
            raise ValueError("Divisão por zero: f(x1) e f(x0) são iguais.")
            
        # Fórmula da Secante
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        
        # Critério de paragem
        if abs(x2 - x1) < tolerancia:
            return x2
            
        # Atualiza os pontos para a próxima iteração: 
        # O ponto x1 antigo passa a ser o x0, e o novo x2 passa a ser o x1
        x0 = x1
        x1 = x2
        
    raise ValueError("O método não convergiu no número máximo de iterações.")

# Exemplo de utilização:
# f = lambda x: x**2 - 4
# print(secante(f, x0=4, x1=5))


def secante_scipy(f, x0):
    # Passamos apenas a função e uma estimativa inicial. 
    # A SciPy trata de gerar o segundo ponto e usar a Secante.
    return optimize.newton(f, x0=x0)


# Halley

def halley(f, df, d2f, x0, tolerancia=1e-6, max_iteracoes=100):
    """
    Encontra a raiz de f(x) usando o Método de Halley (convergência cúbica).
    
    Parâmetros:
    f  : Função cuja raiz queremos encontrar
    df : Primeira derivada analítica
    d2f: Segunda derivada analítica
    x0 : Estimativa inicial
    """
    x = x0
    
    for i in range(max_iteracoes):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)
        
        if dfx == 0:
            raise ValueError(f"A primeira derivada zerou em x = {x}. O método falhou.")
            
        # Calcula o denominador especial da fórmula de Halley
        denominador = dfx - (fx * d2fx) / (2.0 * dfx)
        
        if denominador == 0:
            raise ValueError(f"O denominador de Halley zerou em x = {x}. Divisão por zero.")
            
        # Calcula a próxima aproximação
        proximo_x = x - (fx / denominador)
        
        # Critério de paragem
        if abs(proximo_x - x) < tolerancia:
            return proximo_x
            
        x = proximo_x
        
    raise ValueError("O método não convergiu no número máximo de iterações.")

# Exemplo de utilização (Encontrar raiz de x^3 - 2x - 5 = 0):
# f = lambda x: x**3 - 2*x - 5
# df = lambda x: 3*x**2 - 2
# d2f = lambda x: 6*x
# print(halley(f, df, d2f, x0=2.0))


def halley_scipy(f, df, d2f, x0):
    # Ao fornecer a fprime e a fprime2, ativamos o Halley na SciPy
    return optimize.newton(f, x0=x0, fprime=df, fprime2=d2f)


# Muller


def muller(f, x0, x1, x2, tolerancia=1e-5, max_iteracoes=100):
    """
    Encontra a raiz de f(x) usando o Método de Muller (interpolação parabólica).
    Pode retornar raízes complexas.
    
    Parâmetros:
    f  : Função cuja raiz queremos encontrar
    x0, x1, x2 : Três estimativas iniciais
    """
    for i in range(max_iteracoes):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        
        # Distâncias entre os pontos x
        h0 = x1 - x0
        h1 = x2 - x1
        
        # Diferenças divididas de primeira ordem
        delta0 = (f1 - f0) / h0
        delta1 = (f2 - f1) / h1
        
        # Coeficientes da parábola: a*(x - x2)^2 + b*(x - x2) + c
        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f2
        
        # O discriminante (delta) da fórmula resolvente
        discriminante = cmath.sqrt(b**2 - 4*a*c)
        
        # Escolhemos o sinal no denominador que produz o maior valor em módulo.
        # Isto garante que encontramos a raiz da parábola mais próxima de x2.
        if abs(b + discriminante) > abs(b - discriminante):
            denominador = b + discriminante
        else:
            denominador = b - discriminante
            
        # Calcula o passo de x2 para a nova estimativa x3
        dx = -2 * c / denominador
        x3 = x2 + dx
        
        # Critério de paragem
        if abs(dx) < tolerancia:
            # Retorna apenas a parte real se o imaginário for irrelevante (por questões de formatação)
            if abs(x3.imag) < tolerancia:
                return x3.real
            return x3
            
        # Desloca os pontos para a próxima iteração
        x0, x1, x2 = x1, x2, x3
        
    raise ValueError("O método de Muller não convergiu.")

# Exemplo de utilização (A raiz real de x^3 - 2x - 5 é ~2.0945):
# f = lambda x: x**3 - 2*x - 5
# print(muller(f, x0=1.0, x1=1.5, x2=3.0))


# Método Misto (Brent)

def brent(f, a, b, tolerancia=1e-6, max_iteracoes=100):
    """
    Encontra a raiz de f(x) no intervalo [a, b] usando o Método de Brent.
    Combina Bissecção, Secante e Interpolação Quadrática Inversa.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("A função deve ter sinais opostos em a e b.")

    # Queremos que 'b' seja sempre a nossa melhor estimativa (mais próxima de 0)
    if abs(f(a)) < abs(f(b)):
        a, b = b, a

    c = a
    d = c  # Variável auxiliar para a iteração anterior
    bisseccao_usada = True

    for i in range(max_iteracoes):
        fa, fb, fc = f(a), f(b), f(c)

        # Se encontrámos a raiz exata ou a tolerância foi atingida
        if fb == 0 or abs(b - a) < tolerancia:
            return b

        # Tenta usar um método rápido (IQI ou Secante)
        if fa != fc and fb != fc:
            # Interpolação Quadrática Inversa (3 pontos distintos)
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Método da Secante (se tivermos apenas 2 valores de função distintos)
            s = b - fb * (b - a) / (fb - fa)

        # Condições para rejeitar o método rápido e forçar a Bissecção
        # 1. 's' não está entre (3a + b)/4 e b
        condicao1 = (s < (3 * a + b) / 4) or (s > b)
        # 2 e 3. A convergência está muito lenta (o passo atual não é metade do anterior)
        condicao2 = bisseccao_usada and (abs(s - b) >= abs(b - c) / 2)
        condicao3 = not bisseccao_usada and (abs(s - b) >= abs(c - d) / 2)
        # 4 e 5. A diferença é demasiado pequena
        condicao4 = bisseccao_usada and (abs(b - c) < tolerancia)
        condicao5 = not bisseccao_usada and (abs(c - d) < tolerancia)

        if condicao1 or condicao2 or condicao3 or condicao4 or condicao5:
            # Força a Bissecção como fallback de segurança
            s = (a + b) / 2.0
            bisseccao_usada = True
        else:
            bisseccao_usada = False

        d = c
        c = b

        # Atualiza o intervalo [a, b] para a próxima iteração
        fs = f(s)
        if fa * fs < 0:
            b = s
        else:
            a = s

        # Garante que 'b' continua a ser a melhor estimativa
        if abs(f(a)) < abs(f(b)):
            a, b = b, a

    raise ValueError("O método de Brent não convergiu.")

# Exemplo de utilização:
# f = lambda x: x**3 - 2*x - 5
# print(brent(f, a=2, b=3))


def brent_scipy(f, a, b):
    # O aclamado optimize.brentq (o 'q' vem de quadratic interpolation)
    return optimize.brentq(f, a, b)