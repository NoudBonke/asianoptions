#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Dense>


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

double mean(const double * res, const int N){
    double mean = 0;
    for (auto i=0; i<N; ++i){
        mean += res[i];
    }
    return mean/N;
}

double stdev(const double * res, const int N){
    double var = 0;
    double mu = mean(res, N);
    for (auto i=0; i<N; ++i){
        var += std::pow(res[i] - mu, 2);
    }
    return std::sqrt(var/(N-1));
}

double covar(const double * res1, const double * res2, const double mu2, const int N){
    double var = 0;
    double mu1 = mean(res1, N);
    for (auto i=0; i<N; ++i){
        var += (res1[i] - mu1)*(res2[i]-mu2);
    }
    return var/(N-1);
}


double browniancov(const int N, const double T, const double sigma){
    Eigen::EigenSolver<Eigen::MatrixXd> es;
    Eigen::MatrixXd covariancematrix(N,N);
    double variance = std::pow(sigma, 2);
    for (auto i=0; i<N; ++i){
        for (auto j=0; j<N; ++j){
            // covariance of BM
            covariancematrix(i, j) = variance*std::min((i+1)*T/N,(j+1)*T/N);
        }
    }
    es.compute(covariancematrix, /* computeEigenvectors = */ true);
    //std::cout << "The eigenvalues of A are: " << es.eigenvalues().transpose() << std::endl;
    // solver always returns complex type eigenvalues but matrix is symmetric so imaginary part is zero
    return es.eigenvalues()(0).real(); // PLACEHOLDER, SHOULD RETURN VARIANCE
}

void monte_carlo(double r, double T, double K, double S0, double sigma,  const int N,  const int N_sims, double * res, const double cvexpected, const double cvvar){
    double S[N];
    double cv[N_sims];
    for (auto j=0; j<N_sims; ++j){
        gbm(S0, r, sigma, T, N, S);
        // collect terminal prices for control variates
        cv[j] = S[N];
        res[j] = std::exp(-r*T)*std::max(mean(S, N) - K, zero);
    }
    // control variates based on terminal price

    double cov = covar(res, cv, cvexpected, N_sims);
    double bstar = cov/cvvar;

    double newres[N_sims];
    for (auto i=0; i<N_sims; ++i){
        newres[i] = res[i] - bstar*(cv[i]-cvexpected);
    }
    double avg = mean(newres, N_sims);
    double error = stdev(newres, N_sims)/std::sqrt(N_sims);
    double corr = cov/(sqrt(cvvar)*stdev(res, N_sims));
    std::cout << 1 - std::pow(corr, 2) << std::endl;
    std::cout << "N = " << N << " : " << avg << " +- " << error << std::endl;
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

    // expected final price
    double expectedfinalprice = S0*std::exp(r*T);

    // final price variance
    double varfinalprice = (std::pow(S0,2))*std::exp(2*r*T)*(std::exp(std::pow(sigma, 2)*T)-1);

    // number of simulations
    int N_sims = 10000;
    int N = 256;


    double bc = browniancov(N, T, sigma);
    std::cout << bc << std::endl;
    double res[N_sims];
    monte_carlo(r, T, K, S0, sigma, N, N_sims, res, expectedfinalprice, varfinalprice);
    //writeFile << N << ' ' << res << std::endl;
    writeFile.close();
    return 0;
}


