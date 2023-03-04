#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>

int seed = 42;
double zero = 0.0;
std::normal_distribution<double> distribution_normal(0.0, 1.0);
std::default_random_engine generator(seed);

// geometric brownian motion
// array is passed by pointer
void gbm(const double S0, const double r, const double sigma, const double T, const int N, double * S){
    double dt = T/N;
    // S has length N+1
    S[0] = S0;
    for (auto i=1; i<N+1; ++i){
        S[i] = S[i - 1] * std::exp((r - 0.5 * std::pow(sigma, 2)) * dt + sigma * std::sqrt(dt) * distribution_normal(generator));
    }
    return;
}

double f(const double r, const double T, const double K, const double S0, const double sigma, const int N, double * S){
    gbm(S0, r, sigma, T, N, S);
    double res = 0;
    res += S[0]; res += S[N];
    for (auto i=1; i<N; ++i){
        res += 2*S[i];
    }
    return std::exp(-r*T)*std::max(0.5*res/(N+1) - K, zero);
}


void monte_carlo(double r, double T, double K, double S0, double sigma, int * N, int N_length, int N_sims, double * res){
    double S[N[N_length-1]];
    double S_var[N_sims];
    for(auto i=0; i<N_length; ++i){
        for (auto j=0; j<N_sims; ++j){
            double interm = f(r, T, K, S0, sigma, N[i], S);
            res[i] += interm;
            S_var[j] = interm;
        }
        res[i] /= N_sims;
        double var = 0;
        for(auto j=0; j<N_sims; ++j){
            var += std::pow(S_var[j] - res[i], 2);
        }
        var /= N_sims - 1;
        std::cout << "N = " << N[i] << " : " << res[i] << " +- " << std::sqrt(var)/std::sqrt(N_sims) << std::endl;
    }
    return;
}
int main(int argc, char* argv[]){

    std::ofstream writeFile;
    writeFile.open("output.txt");
    if (!writeFile.is_open()){
        std::cout << "File not found\n";
        return 0;
    }
    // volatility
    double sigma = 0.25;

    // maturity
    double T = 3; // years

    // short rate
    double r = 0.04;

    // strike price
    double K = 100;

    // initial stock price
    double S0 = 100;

    // number of simulations
    int N_sims = 10000;

    // const int N_length = 76;
    // int N[N_length];
    // double res[N_length];
    // for(auto k=0; k<N_length-1; ++k){
    //     N[k] = 10*(k+1);
    //     res[k] = 0;
    // }
    // N[N_length-1] = 10000;
    // res[N_length-1] = 0;
    int N_length = 4;
    int N[N_length] = {6, 36, 150, 750};
    double res[N_length] = {0, 0, 0, 0};
    monte_carlo(r, T, K, S0, sigma, N, N_length, N_sims, res);
    for(auto k=0; k<N_length; ++k){
        writeFile << N[k] << ' ' << res[k] << std::endl;
    }
    writeFile.close();
    return 0;
}


