#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/Eigen/Dense>


int seed = 20;
double zero = 0.0;
std::normal_distribution<double> distribution_normal(0.0, 1.0);
std::default_random_engine generator(seed);

// geometric brownian motion
// array is passed by pointer
void gbm(const double S0, const double r, const double sigma, const double T, const int N, double * S){
    double dt = T/(N-1);
    // S has length N+1
    S[0] = S0;
    for (auto i=1; i<N; ++i){
        S[i] = S[i - 1] * std::exp((r - 0.5 * std::pow(sigma, 2)) * dt + sigma * std::sqrt(dt) * distribution_normal(generator));
    }
    return;
}

double gbm_PCA(const double S0, const double r, const double sigma, const double T, const int N, double * S, Eigen::MatrixXd &PCA){
    double dt = T/(N-1);
    S[0] = S0;
    Eigen::VectorXd normals(N-1);
    for (auto i=0; i<N-1; ++i){
        normals[i] = distribution_normal(generator);
    }
    Eigen::VectorXd PCAresultEigen = PCA*normals;
    //std::cout << PCAresultEigen << std::endl;
    for (auto i=0; i<N-1; ++i){
        S[i+1] = (r - std::pow(sigma, 2)/2)*i*dt + PCAresultEigen[i];
        S[i+1] = S0*std::exp(S[i+1]);
    }
    // control variate
    return normals[0];
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


void browniancov(const int N, const double T, const double sigma, Eigen::MatrixXd &PCA){
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
    for (auto i=0; i<N; ++i){
        PCA.col(i) = (std::sqrt(es.eigenvalues()[i])*es.eigenvectors().col(i).normalized()).real();
    }
    return;
}

// void monte_carlo_terminal(double * S, double r, double T, double K, double S0, double sigma,  const int N,  const int N_sims, Eigen::MatrixXd &PCA, double * res, const double cvexpected, const double cvvar){
//     double S[N];
//     double cv[N_sims];
//     for (auto j=0; j<N_sims; ++j){
//         double dummy = gbm_PCA(S0, r, sigma, T, N, S, PCA);
//         // collect terminal prices for control variates
//         cv[j] = S[N-1];
//         res[j] = std::exp(-r*T)*std::max(mean(S, N) - K, zero);
//     }
//     // control variates based on terminal price

//     double cov = covar(res, cv, cvexpected, N_sims);
//     double bstar = cov/cvvar;

//     double newres[N_sims];
//     for (auto i=0; i<N_sims; ++i){
//         newres[i] = res[i] - bstar*(cv[i]-cvexpected);
//     }
//     double avg = mean(newres, N_sims);
//     double error = stdev(newres, N_sims)/std::sqrt(N_sims);
//     double corr = cov/(sqrt(cvvar)*stdev(res, N_sims));
//     std::cout << "no variance reduction: " << std::endl;
//     std::cout << "N = " << N << " : " << mean(res, N_sims) << " +- " << stdev(res, N_sims)/std::sqrt(N_sims) << std::endl;
//     std::cout << "variance reduction: " << 1 - std::pow(corr, 2) << std::endl;
//     std::cout << "N = " << N << " : " << avg << " +- " << error << std::endl;
//     return;
// }

void monte_carlo_PCA(double r, double T, double K, double S0, double sigma,  const int N,  const int N_sims, Eigen::MatrixXd &PCA, double * res, const double cv_terminal_expected, const double cv_terminal_var, const double cv_PCA_expected, const double cv_PCA_var){
    double S[N];
    double cv_PCA[N_sims];
    double cv_terminal[N_sims];
    for (auto j=0; j<N_sims; ++j){
        cv_PCA[j] = gbm_PCA(S0, r, sigma, T, N, S, PCA);
        cv_terminal[j] = S[N-1];
        res[j] = std::exp(-r*T)*std::max(mean(S, N) - K, zero);
    }
    // control variates based on terminal price

    double cov_PCA = covar(res, cv_PCA, cv_PCA_expected, N_sims);
    double bstar_PCA = cov_PCA/cv_PCA_var;

    double cov_terminal = covar(res, cv_terminal, cv_terminal_expected, N_sims);
    double bstar_terminal = cov_terminal/cv_terminal_var;

    double newres_PCA[N_sims];
    double newres_terminal[N_sims];

    for (auto i=0; i<N_sims; ++i){
        newres_PCA[i] = res[i] - bstar_PCA*(cv_PCA[i]-cv_PCA_expected);
        newres_terminal[i] = res[i] - bstar_terminal*(cv_terminal[i]-cv_terminal_expected);
    }
    double avg_PCA = mean(newres_PCA, N_sims);
    double error_PCA = stdev(newres_PCA, N_sims)/std::sqrt(N_sims);
    double corr_PCA = cov_PCA/(sqrt(cv_PCA_var)*stdev(res, N_sims));
    double avg_terminal = mean(newres_terminal, N_sims);
    double error_terminal = stdev(newres_terminal, N_sims)/std::sqrt(N_sims);
    double corr_terminal = cov_terminal/(sqrt(cv_terminal_var)*stdev(res, N_sims));
    std::cout << "no variance reduction: " << std::endl;
    std::cout << "N = " << N << " : " << mean(res, N_sims) << " +- " << stdev(res, N_sims)/std::sqrt(N_sims) << std::endl;
    std::cout << "PCA control variate variance reduction: " << 1 - std::pow(corr_PCA, 2) << std::endl;
    std::cout << "N = " << N << " : " << avg_PCA << " +- " << error_PCA << std::endl;
    std::cout << "terminal price control variate variance reduction: " << 1 - std::pow(corr_terminal, 2) << std::endl;
    std::cout << "N = " << N << " : " << avg_terminal << " +- " << error_terminal << std::endl;
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
    //int N_sims = std::pow(10, 4);
    int N = 256;

    //first price is fixed at S0 so only 255 values may vary.
    Eigen::MatrixXd PCA(N-1,N-1);
    browniancov(N-1, T, sigma, PCA);

    for (auto j=3; j<6; ++j){
        int N_sims = std::pow(10, j);
        double res[N_sims];
        std::cout << "\n# of iterations : " << N_sims << std::endl;
        //std::cout << "\nControl variate: first PCA component" << std::endl;
        monte_carlo_PCA(r, T, K, S0, sigma, N, N_sims, PCA, res, expectedfinalprice, varfinalprice, 0, 1);
        //std::cout << "Control variate: terminal price" << std::endl;
        //monte_carlo_PCA(r, T, K, S0, sigma, N, N_sims, PCA, res, 0, 1);
    }

    //writeFile << N << ' ' << res << std::endl;
    writeFile.close();
    return 0;
}


