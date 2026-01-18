import numpy as np
import time
from fpylll import IntegerMatrix, LLL, BKZ, GSO

class LWEInstance:
    """
    Klasa pomocnicza przechowująca instancję problemu LWE.
    """
    def __init__(self, n, m, q, alpha):
        self.n = n          # Wymiar sekretu
        self.m = m          # Liczba próbek
        self.q = q          # Moduł
        self.alpha = alpha  # Odchylenie standardowe szumu
        
        self.A = None
        self.s = None
        self.e = None
        self.b = None

    def generate(self):
        """Generuje losową instancję LWE: b = As + e mod q."""
        print(f"[*] Generowanie instancji LWE (n={self.n}, m={self.m}, q={self.q})...")
        
        # 1. Macierz publiczna A (m x n)
        self.A = np.random.randint(0, self.q, size=(self.m, self.n))
        
        # 2. Sekret s (n) - losowy jednostajny
        self.s = np.random.randint(0, self.q, size=self.n)
        
        # 3. Błąd e (m) - rozkład Gaussa, zaokrąglony do liczb całkowitych
        sigma = self.alpha * self.q
        # W praktyce LWE często używa dyskretnego rozkładu Gaussa, tutaj przybliżenie:
        self.e = np.round(np.random.normal(0, sigma, size=self.m)).astype(int)
        
        # 4. Obliczenie b = As + e (mod q)
        # Używamy np.dot dla mnożenia macierzy
        self.b = (np.dot(self.A, self.s) + self.e) % self.q
        
        print("[+] Instancja wygenerowana pomyślnie.")
        return self.A, self.b

    def check_solution(self, candidate_s):
        """Weryfikuje, czy podany kandydat na sekret jest poprawny."""
        if candidate_s is None:
            return False
            
        # Sprawdź czy As' jest blisko b (różnica powinna być małym błędem)
        b_calc = np.dot(self.A, candidate_s) % self.q
        diff = (self.b - b_calc + self.q // 2) % self.q - self.q // 2 # Centrowanie modulo
        
        # Jeśli norma różnicy jest mała, rozwiązanie jest poprawne
        norm_diff = np.linalg.norm(diff)
        expected_norm = np.sqrt(self.m) * (self.alpha * self.q)
        
        # Margines błędu dla weryfikacji (np. 1.5x oczekiwanej normy)
        is_correct = norm_diff < (expected_norm * 2.0)
        
        # Dodatkowa weryfikacja dokładna (dla symulacji, gdy znamy s)
        is_exact = np.array_equal(candidate_s, self.s)
        
        return is_correct or is_exact

def build_primal_lattice(A, b, q):
    """
    Konstruuje macierz bazy dla ataku Primal (Kannan's Embedding).
    
    B = [ qI_m   0     0 ]
        [  A^T   I_n   0 ]
        [  b^T   0     1 ]
    """
    m, n = A.shape
    d = m + n + 1
    
    # Tworzymy macierz w fpylll (IntegerMatrix)
    B = IntegerMatrix(d, d)
    
    # Wypełnianie bloku q*I_m (lewy górny róg)
    for i in range(m):
        B[i, i] = q
        
    # Wypełnianie bloku A^T (lewy środek) i I_n (prawy środek)
    # A ma wymiary m x n, więc A^T ma n x m.
    # W fpylll indeksujemy B[wiersz, kolumna]
    for i in range(n):
        row_idx = m + i
        # Wstawianie wiersza z A^T (czyli kolumny z A)
        for j in range(m):
            B[row_idx, j] = int(A[j, i]) # A[j, i] to element A^T[i, j]
        
        # Wstawianie I_n (diagonalnie obok A^T)
        B[row_idx, m + i] = 1
        
    # Wypełnianie wektora b (ostatni wiersz)
    row_last = m + n
    for j in range(m):
        B[row_last, j] = int(b[j])
        
    # Wstawianie 1 na końcu (embedding factor)
    B[row_last, m + n] = 1
    
    return B

def primal_attack(A, b, q, block_size=20):
    """
    Przeprowadza atak Primal używając redukcji BKZ.
    """
    m, n = A.shape
    print(f"[*] Rozpoczynanie ataku Primal.")
    print(f"[*] Wymiar kraty: {m + n + 1} x {m + n + 1}")
    
    # 1. Konstrukcja kraty
    start_time = time.time()
    B = build_primal_lattice(A, b, q)
    print(f"[+] Krata skonstruowana.")
    
    # 2. Redukcja LLL (wstępna)
    print(f"[*] Uruchamianie LLL...")
    LLL.reduction(B)
    
    # 3. Redukcja BKZ (silniejsza)
    print(f"[*] Uruchamianie BKZ-{block_size}...")
    BKZ.reduction(B, BKZ.Param(block_size))
    
    reduction_time = time.time() - start_time
    print(f"[+] Redukcja zakończona w {reduction_time:.2f}s.")
    
    # 4. Poszukiwanie sekretu w zredukowanej bazie
    # Spodziewamy się wektora v = (e, s, 1) lub v = (-e, -s, -1)
    # Sekret s znajduje się na indeksach [m : m+n]
    
    # Sprawdzamy kilka pierwszych najkrótszych wektorów
    for row_idx in range(min(10, B.nrows)):
        v = B[row_idx]
        
        # Pobierz potencjalny sekret (kandydat)
        # W fpylll wektory są indeksowane jak listy
        s_candidate = [v[j] for j in range(m, m + n)]
        s_candidate = np.array(s_candidate)
        
        # Sprawdzamy s
        # Musimy sprawdzić, czy działa dla oryginalnego równania modulo q
        # Z uwagi na modulo, ujemne wartości w Pythonie trzeba traktować ostrożnie,
        # ale tutaj s jest elementem Z_q, więc zrobimy modulo.
        
        s_cand_mod = s_candidate % q
        if verify_candidate(A, b, s_cand_mod, q):
             return s_cand_mod, reduction_time

        # Sprawdzamy -s (ponieważ SVP jest z dokładnością do znaku)
        s_cand_neg_mod = (-s_candidate) % q
        if verify_candidate(A, b, s_cand_neg_mod, q):
             return s_cand_neg_mod, reduction_time
             
    print("[-] Atak nie powiódł się. Spróbuj zwiększyć block_size lub liczbę próbek m.")
    return None, reduction_time

def verify_candidate(A, b, s_cand, q, alpha=0.005): # Dodaj alpha
    b_prime = np.dot(A, s_cand) % q
    diff = (b - b_prime + q//2) % q - q//2
    norm = np.linalg.norm(diff)
    
    # Obliczamy oczekiwaną normę dla PRAWDZIWEGO sekretu
    # m = len(b)
    expected_norm = alpha * q * np.sqrt(len(b))
    
    # Dajemy mały margines (np. 1.5x lub 2x), ale nie q/4!
    limit = expected_norm * 2.0
    
    return norm < limit

# --- GŁÓWNA FUNKCJA ---

if __name__ == "__main__":
    print("=========================================")
    print("   PROJEKT: ATAK PRIMAL NA LWE (DEMO)    ")
    print("=========================================")
    
    # PARAMETRY "TOY EXAMPLE" (Małe, aby działało szybko na laptopie)
    # W prawdziwym LWE n=500+, tutaj n=40
    n = 10
    m = 60          # Zazwyczaj m = 2n dla wystarczającej nadmiarowości
    q = 101         # Liczba pierwsza
    alpha = 0.001   # Parametr szumu (sigma = alpha * q)
    
    # Parametr ataku
    bkz_block_size = 25 # Im większy, tym silniejszy atak, ale wolniejszy
    
    # 1. Generowanie Wyzwania
    lwe = LWEInstance(n, m, q, alpha)
    A, b = lwe.generate()
    
    print(f"\nSekret do znalezienia (pierwsze 5 el.): {lwe.s[:5]}...")
    
    # 2. Uruchomienie ataku
    recovered_s, time_taken = primal_attack(A, b, q, block_size=bkz_block_size)
    
    # 3. Wyniki
    print("\n=========================================")
    print("                WYNIKI                   ")
    print("=========================================")
    
    if recovered_s is not None:
        print("[SUKCES] Znaleziono sekret!")
        print(f"Oryginalny s:   {lwe.s}")
        print(f"Odzyskany s:    {recovered_s}")
        
        # Ostateczna weryfikacja
        if np.array_equal(lwe.s, recovered_s):
            print("Weryfikacja: POPRAWNY (Idealna zgodność)")
        else:
            print("Weryfikacja: POPRAWNY (Matematycznie równoważny)")
    else:
        print("[PORAŻKA] Nie udało się odzyskać sekretu.")
        print("Sugestie:")
        print(" - Zwiększ parametr block_size w BKZ")
        print(" - Zwiększ liczbę próbek m")
        print(" - Zmniejsz szum (alpha)")
        
    print("=========================================")