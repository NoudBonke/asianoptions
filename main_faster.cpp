#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>

int seed = 42;
double zero = 0.0;
std::normal_distribution<float> distribution_normal(0.0, 1.0);
std::default_random_engine generator(seed);

// geometric brownian motion
// array is passed by pointer
void gbm(const float S0, const float r, const float sigma, const float T, const int N, float * S){
    float dt = T/N;
    // S has length N+1
    S[0] = S0;
    for (auto i=1; i<N+1; ++i){
        S[i] = S[i - 1] * std::exp((r - 0.5 * std::pow(sigma, 2)) * dt + sigma * std::sqrt(dt) * distribution_normal(generator));
    }
    return;
}

float f(const float r, const float T, const float K, const float S0, const float sigma, const int N, float * S){
    gbm(S0, r, sigma, T, N, S);
    float res = 0;
    res += S[0]; res += S[N];
    for (auto i=1; i<N; ++i){
        res += 2*S[i];
    }
    return std::exp(-r*T)*std::max(0.5*res/N - K, zero);
}


void monte_carlo(float r, float T, float K, float S0, float sigma, int * N, int N_length, int N_sims, float * res){
    float S[N[N_length-1]];
    for(auto i=0; i<N_length; ++i){
        for (auto j=0; j<N_sims; ++j){
            res[i] += f(r, T, K, S0, sigma, N[i], S);
        }
        res[i] /= N_sims;
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
    float sigma = 0.25;

    // maturity
    float T = 3; // years

    // short rate
    float r = 0.04;

    // strike price
    float K = 100;

    // initial stock price
    float S0 = 100;

    // number of simulations
    int N_sims = 1000;

    const int N_length = 75;
    int N[N_length];
    float res[N_length];
    for(auto k=0; k<N_length; ++k){
        N[k] = 10*(k+1);
        res[k] = 0;
    }
    monte_carlo(r, T, K, S0, sigma, N, N_length, N_sims, res);
    for(auto k=0; k<N_length; ++k){
        writeFile << N[k] << ' ' << res[k] << std::endl;
    }
    writeFile.close();
    return 0;
}


