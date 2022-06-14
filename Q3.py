from math import sqrt, pi

x = 1
i = 0
while x > 0:
    x = x/2
    i += 1
mach_epsilon = 1/2**(i-1)

def isprime(n):
    """Returns True if n is prime."""
    if n == 2:
        return True
    if n == 3:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False

    i = 5
    w = 2

    while i * i <= n:
        if n % i == 0:
            return False

        i += w
        w = 6 - w

    return True

def no_diviser(x1, x2):
    # Checks if x2 divides x1
    out = x1/x2
    if out-int(out) != 0:
        return True
    return False

def no_con_with_coeffs(used_coeffs, num):
    for coeff in used_coeffs:
        # roudng because 1/num and 1/coeff must be intergers 
        if not no_diviser(round(1/num), round(1/coeff)):
            return False
    return True

def next_prime(coeff, prime=isprime):
    """Given 1/p^2 where p is prime
    finds the next prime number after p,
    say q, and returns 1/q^2"""
    i = 1
    denom = round(1/sqrt(coeff))
    next = denom + i
    while not prime(next):
        i += 1
        next = denom + i
    return 1/next**2


def s2(epsilon=mach_epsilon, isprime=isprime, coprime=no_diviser, \
          no_con_with_coeffs=no_con_with_coeffs, next_prime=next_prime):
    """Summs up Sigma_{n=1}^{infty} 1/n^2"""
    used_coeffs = []
    i = 2 
    e = 1
    i_terms = 1 # expression starts from 1
    total = 1 # expression starts from 1
    coeff = 1/i**2
    # multiplies the coefficient to the bracket
    # sum and adds to the total 
    while coeff >= epsilon and abs(total - (pi**2)/6)>10**(-8):
        e = 1
        i_terms += 1
        term = 1/i**2
        bracket_sum = 1 # starts from 1 then goes to the coeff value

        # Sums the terms in the bracket until saturation
        while coeff*term >= epsilon:
            if no_con_with_coeffs(used_coeffs, term):
                bracket_sum += term
                i_terms += 1

            term = 1/round((i+e)**2) # rounding because we
                                     # know it must be integer
            e += 1

        total += coeff*bracket_sum
        used_coeffs.append(coeff) # filling the list of used coeffs
        i += 1
        coeff = next_prime(coeff) #gives the next coefficient

    return i_terms, pi**2/6, total, (pi**2/6-total)/1

print(s2(1/1000**2))
