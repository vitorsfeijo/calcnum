import numpy as np

# Derivadas parciais (gradiente)
def grad_f(x, y):
    fx = 12*x**3 + 2*y + 1  # df/dx
    fy = 2*x - 8*y**3 - 1   # df/dy
    return np.array([fx, fy])

# Segunda derivadas (matriz Hessiana)
def hess_f(x, y):
    fxx = 36*x**2
    fxy = 2
    fyy = -24*y**2
    return np.array([[fxx, fxy],
                     [fxy, fyy]])

# Método de Newton
def newton_minimo(x0, y0, tol=1e-8, max_iter=100):
    x = x0
    y = y0
    
    for k in range(max_iter):
        g = grad_f(x, y)
        H = hess_f(x, y)
        
        # Resolvendo o sistema H * delta = gradiente
        delta = np.linalg.inv(H).dot(g)
        
        # Atualizando x e y
        x = x - delta[0]
        y = y - delta[1]
        
        # Critério de parada
        if np.linalg.norm(delta) < tol:
            print(f"Convergiu em {k+1} iterações.")
            return x, y, k+1
            
    print("Não convergiu dentro do número máximo de iterações.")
    return x, y, max_iter

# Execução principal
x0 = -1
y0 = 1
tol = 1e-8

x_min, y_min, iteracoes = newton_minimo(x0, y0, tol)
f_min = 3*x_min**4 + 2*x_min*y_min - 2*y_min**4 + x_min - y_min

print("\n--- Resultado final ---")
print(f"Ponto de mínimo aproximado: x = {x_min:.10f}, y = {y_min:.10f}")
print(f"Iterações necessárias: {iteracoes}")