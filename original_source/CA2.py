import numpy as np
import matplotlib.pyplot as plt

def MetroGauss(b=1, x=0, e=1, arate = True, T=100000, Therm=5000):
    xs = np.zeros(T)  # Array to store sampled values
    accepted = 0  # Counter for accepted moves to track acceptance rate

    for i in range(T):
        
        delta = 2 * e * (np.random.rand() - 0.5)   # Random trial move 
        xprime = x + delta  # Propose a new position        
        dH = xprime**2 - x**2
        P = min(1, np.exp(-b * dH))  # Compute the Metropolis acceptance probability
        
        if np.random.rand() <= P: # Accept or reject the proposed move based on probability P
            x = xprime  # Accept move
            accepted += 1  # Increase accepted move counter
        
        xs[i] = x  # Store the current position in the array

    acceptance_rate = accepted / T  # Compute acceptance rate
    if arate == True:
        print(f"Acceptance Rate: {acceptance_rate:.2f}")

    samples = xs[Therm:]    # Discard initial thermalization steps to obtain final samples

    plt.hist(samples, bins=100, density=True, alpha=0.7, color='b', edgecolor='black')
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title(f"Metropolis Sampling at β = {b}")
    plt.show()

    return samples  # Return the thermalized samples

#BinAnalyse computes the expectation value of F as well as the binned error, but takes F(x) as an input rather than just x itself
def BinAnalyse(samples, binsize=20): 
    L = len(samples) // binsize  # Compute number of bins
    binList = [np.mean(samples[i*binsize:(i+1)*binsize]) for i in range(L)]  # Splits samples into binsize, computes the mean of each bin, stores the mean in binlist.
    
    return np.mean(samples),  np.std(binList) / np.sqrt(L-1)   # Compute mean and error using binning.

def AutoCorrelation(samples, max_lag=100):
    mean_x = np.mean(samples)  # Compute mean of sampled values
    var_x = np.var(samples)  # Compute variance

    autocorr = np.correlate(samples - mean_x, samples - mean_x, mode='full')    # Compute full autocorrelation function
    autocorr = autocorr[len(samples)-1:] / (var_x * len(samples))  # Normalize

    # Plot autocorrelation function
    plt.figure(figsize=(8, 4))
    plt.plot(autocorr[:max_lag], marker='.', linestyle='-', color='r')
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Function")
    plt.grid()
    plt.show()

    return autocorr[:max_lag]  # Return autocorrelation values


def Reweight(B_new, B_old=1, x=None):
    if x is None:
        raise ValueError("Sampled data x is required for reweighting.")

    delB = B_new - B_old  # Compute difference in β
    numO = x**2 * np.exp(-delB * x**2)  # Numerator: weighted observable
    denO = np.exp(-delB * x**2)  # Denominator: partition function adjustment
    return np.mean(numO) / np.mean(denO)  # Compute reweighted estimate

beta_original = 1 # Set initial inverse temperature β = 1
Xs = MetroGauss(b=beta_original, e=1.0, T=100000, Therm=5000) # Run Metropolis algorithm and generate sampled values

# Compute direct measurement of energy from sampled values
E_direct, E_error = BinAnalyse(Xs**2)
print(f"Expectation Value at β = 1: <E> = {E_direct:.4f} ± {E_error:.4f}")

print('\nCompare reweighted estimates with direct sampling at different values of β:')

# Compare reweighted estimates at different β values
B_values = [0.5, 1.5, 2.0] # Define β values for reweighting
e_values = [1.3, 0.8, 0.7]  # Define e values for direct measurment at β values (so that the acceptance rate is approximately equal to that at β = 1)
print("Reweighted Energy Estimates:")
    
for B_new in B_values:
    E_reweight = Reweight(B_new, x=Xs)
    print(f"Reweighted <E> at β = {B_new}: {E_reweight:.4f}")

print("\nDirect Energy Measurements:")
print("\nDirect Sampling:")

for b,e in zip(B_values,e_values): # Measure expectation value directly at β
    X_dir = MetroGauss(b,0,e, arate = False) #sample at β
    EExp = BinAnalyse(X_dir**2) # Extract expectation and error
    E_dir, E_err = EExp[0], EExp[1]
    print(f"Direct <E> at β = {b}: = {E_dir:.4f} ± {E_err:.4f}")
    
# Compute and visualize autocorrelation function to analyze sample correlations
AutoCorrelation(Xs)