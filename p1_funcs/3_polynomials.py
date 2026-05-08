import numpy as np

# Regras de Inspeção (Descartes, Du Gua e Lacuna)

def regras_inspecao(coeficientes):
    """
    Analisa um polinômio usando as regras de Descartes, Du Gua e Lacunas.
    coeficientes: Lista ordenada do maior para o menor grau. Ex: x^3 - 2x - 5 -> [1, 0, -2, -5]
    """
    resultados = {}
    
    # ---------------------------------------------------------
    # 1. Regra de Descartes (Sinais)
    # ---------------------------------------------------------
    def contar_trocas_sinal(coefs):
        # Ignora os zeros para contar as trocas
        c_sem_zero = [c for c in coefs if c != 0]
        trocas = sum(1 for i in range(len(c_sem_zero)-1) if c_sem_zero[i] * c_sem_zero[i+1] < 0)
        return trocas

    # Máximo de raízes positivas = trocas de sinal em P(x)
    max_positivas = contar_trocas_sinal(coeficientes)
    
    # Máximo de raízes negativas = trocas de sinal em P(-x)
    # P(-x) inverte o sinal dos coeficientes das potências ímpares
    grau_maximo = len(coeficientes) - 1
    coefs_negativos = []
    for i, c in enumerate(coeficientes):
        potencia = grau_maximo - i
        if potencia % 2 != 0: # É ímpar
            coefs_negativos.append(-c)
        else:                 # É par
            coefs_negativos.append(c)
            
    max_negativas = contar_trocas_sinal(coefs_negativos)
    
    resultados['descartes'] = {
        'max_raizes_positivas': max_positivas,
        'max_raizes_negativas': max_negativas
    }

    # ---------------------------------------------------------
    # 2. Regra da Lacuna e Regra de Du Gua
    # ---------------------------------------------------------
    tem_lacuna = False
    du_gua_complexas = False
    detalhes_du_gua = []
    
    # Verificamos apenas os coeficientes do meio (o primeiro e o último não são lacunas)
    for i in range(1, len(coeficientes) - 1):
        if coeficientes[i] == 0:
            tem_lacuna = True
            
            # Regra de Du Gua: Se os coeficientes vizinhos à lacuna têm o MESMO sinal,
            # então o polinômio possui raízes complexas.
            vizinho_esq = coeficientes[i-1]
            vizinho_dir = coeficientes[i+1]
            
            if vizinho_esq * vizinho_dir > 0:
                du_gua_complexas = True
                grau_lacuna = grau_maximo - i
                detalhes_du_gua.append(
                    f"Lacuna no grau x^{grau_lacuna} cercada por mesmos sinais ({vizinho_esq}, {vizinho_dir})"
                )
                
    resultados['lacunas'] = tem_lacuna
    resultados['du_gua'] = {
        'indica_raizes_complexas': du_gua_complexas,
        'detalhes': detalhes_du_gua
    }

    return resultados

# Exemplo de utilização: P(x) = x^4 + 3x^2 - x - 2
# Coeficientes: [1, 0, 3, -1, -2]
# print(regras_inspecao([1, 0, 3, -1, -2]))


# Cota de Kojima (e Fujiwara/Cauchy)

def cotas_raizes(coeficientes):
    """
    Calcula as cotas superiores para o módulo das raízes de um polinómio.
    coeficientes: Lista do maior grau para o menor. Ex: x^3 - 2x - 5 -> [1, 0, -2, -5]
    """
    # Remove zeros à esquerda (se existirem, não fazem parte do polinómio)
    coefs = list(coeficientes)
    while len(coefs) > 1 and coefs[0] == 0:
        coefs.pop(0)
        
    n = len(coefs) - 1
    if n < 1:
        raise ValueError("O grau do polinómio tem de ser pelo menos 1.")
        
    a_n = abs(coefs[0])
    
    # ---------------------------------------------------------
    # 1. Cota de Cauchy
    # Raio = 1 + max(|a_i / a_n|) para todo i < n
    # ---------------------------------------------------------
    max_cauchy = max(abs(c) / a_n for c in coefs[1:])
    cota_cauchy = 1 + max_cauchy
    
    # ---------------------------------------------------------
    # 2. Cota de Kojima e Fujiwara
    # Baseiam-se na sequência q_i = |a_{n-i} / a_n|^(1/i)
    # ---------------------------------------------------------
    valores_q = []
    for i in range(1, n + 1):
        q_i = (abs(coefs[i]) / a_n) ** (1.0 / i)
        valores_q.append(q_i)
        
    # Cota de Kojima: A soma dos dois maiores valores da sequência 'q'
    valores_q_ordenados = sorted(valores_q, reverse=True)
    if len(valores_q_ordenados) == 1:
        cota_kojima = valores_q_ordenados[0]
    else:
        cota_kojima = valores_q_ordenados[0] + valores_q_ordenados[1]
        
    # Cota de Fujiwara: O dobro do maior valor da sequência 'q'
    # (Algumas versões estritas dividem o último termo por 2 antes do max, 
    # mas esta é a forma mais adotada em cálculo numérico geral)
    cota_fujiwara = 2 * valores_q_ordenados[0]

    return {
        'cauchy': cota_cauchy,
        'kojima': cota_kojima,
        'fujiwara': cota_fujiwara,
        'melhor_cota': min(cota_cauchy, cota_kojima, cota_fujiwara)
    }

# Exemplo de utilização (Polinómio P(x) = x^3 + 2x^2 - 5x - 6):
# Raízes reais exatas: -3, -1, 2 (o módulo máximo é 3)
# print(cotas_raizes([1, 2, -5, -6]))



# Método de Newton-Raphson com Método de Horner

def horner_com_derivada(coeficientes, x):
    """
    Avalia P(x) e a sua derivada P'(x) usando o Método de Horner.
    coeficientes: Lista do maior para o menor grau. Ex: [1, 0, -2, -5] para x^3 - 2x - 5
    """
    if not coeficientes:
        raise ValueError("A lista de coeficientes não pode estar vazia.")
        
    p = coeficientes[0]
    dp = 0.0 # Derivada
    
    # O ciclo começa do segundo coeficiente até ao final
    for c in coeficientes[1:]:
        # A derivada acumula o valor anterior de p multiplicado por x
        dp = dp * x + p
        # O polinómio acumula o valor de p multiplicado por x mais o novo coeficiente
        p = p * x + c
        
    return p, dp

def newton_horner(coeficientes, x0, tolerancia=1e-6, max_iteracoes=100):
    """
    Encontra uma raiz real de um polinómio usando Newton-Raphson otimizado por Horner.
    """
    x = x0
    
    for i in range(max_iteracoes):
        # Avalia o polinómio e a derivada de uma só vez!
        px, dpx = horner_com_derivada(coeficientes, x)
        
        if dpx == 0:
            raise ValueError(f"A derivada zerou em x = {x}. O método falhou.")
            
        # Fórmula clássica de Newton-Raphson
        proximo_x = x - (px / dpx)
        
        # Critério de paragem: o passo é menor que a tolerância ou P(x) é muito próximo de 0
        if abs(proximo_x - x) < tolerancia or abs(px) < tolerancia:
            return proximo_x
            
        x = proximo_x
        
    raise ValueError("O método não convergiu no número máximo de iterações.")

# Exemplo de utilização (Polinómio P(x) = x^3 - 2x - 5):
# A raiz real aproximada é 2.09455
# print(newton_horner([1, 0, -2, -5], x0=3.0))


# Método de Bairstow


import cmath

def bairstow(coeficientes, r0=0.5, s0=-0.5, tolerancia=1e-5, max_iteracoes=100):
    """
    Encontra TODAS as raízes de um polinómio usando o Método de Bairstow.
    coeficientes: Lista do maior grau para o menor. Ex: x^4 - 2x^3 + 4x - 4 -> [1, -2, 0, 4, -4]
    """
    a = list(coeficientes)
    raizes = []

    # O ciclo continua enquanto o grau do polinómio for >= 3 (tamanho >= 4)
    while len(a) > 3:
        n = len(a) - 1
        r, s = r0, s0
        
        for _ in range(max_iteracoes):
            b = [0] * (n + 1)
            c = [0] * (n + 1)
            
            # Divisão sintética para encontrar os coeficientes b
            b[0] = a[0]
            b[1] = a[1] + r * b[0]
            for i in range(2, n + 1):
                b[i] = a[i] + r * b[i-1] + s * b[i-2]
                
            # Divisão sintética nos coeficientes b para encontrar as derivadas (c)
            c[0] = b[0]
            c[1] = b[1] + r * c[0]
            for i in range(2, n): # Para c, só precisamos ir até n-1
                c[i] = b[i] + r * c[i-1] + s * c[i-2]
                
            # Sistema de equações lineares 2x2 (Regra de Cramer)
            det = c[n-2] * c[n-2] - c[n-1] * c[n-3]
            
            if det == 0:
                # Perturbação para evitar divisão por zero se cair num ponto de sela
                r += 0.1
                s += 0.1
                continue
                
            dr = (-b[n-1] * c[n-2] + b[n] * c[n-3]) / det
            ds = (-b[n] * c[n-2] + b[n-1] * c[n-1]) / det
            
            r += dr
            s += ds
            
            # Critério de paragem: os incrementos de r e s são minúsculos
            if abs(dr) < tolerancia and abs(ds) < tolerancia:
                break
                
        # Extrai as duas raízes do fator (x^2 - rx - s = 0)
        # Fórmula: x = (r +- sqrt(r^2 + 4s)) / 2
        delta = cmath.sqrt(r**2 + 4*s)
        x1 = (r + delta) / 2.0
        x2 = (r - delta) / 2.0
        raizes.extend([x1, x2])
        
        # Deflação: o novo polinómio (dividido pelo fator quadrático) 
        # é composto pelos coeficientes 'b' até n-2
        a = b[:-2]
        
    # Quando o grau for 2 (tamanho 3), resolvemos com a fórmula quadrática direta
    if len(a) == 3: 
        delta = cmath.sqrt(a[1]**2 - 4*a[0]*a[2])
        x1 = (-a[1] + delta) / (2*a[0])
        x2 = (-a[1] - delta) / (2*a[0])
        raizes.extend([x1, x2])
        
    # Quando o grau for 1 (tamanho 2), resolvemos a equação linear a[0]x + a[1] = 0
    elif len(a) == 2: 
        raizes.append(-a[1] / a[0])
        
    # Limpeza para retornar reais "limpos" (sem o +0j da parte imaginária)
    raizes_limpas = []
    for raiz in raizes:
        if abs(raiz.imag) < tolerancia:
            raizes_limpas.append(raiz.real)
        else:
            raizes_limpas.append(complex(round(raiz.real, 5), round(raiz.imag, 5)))
            
    return raizes_limpas

# Exemplo de utilização: P(x) = x^4 - 2x^3 + 4x^2 - 4x + 4
# As raízes são: 1+i, 1-i, raiz(2)i, -raiz(2)i
# print(bairstow([1, -2, 4, -4, 4]))


# print(np.roots([1, -2, 4, -4, 4]))