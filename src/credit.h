#ifndef credit_cpp
#define credit_cpp

#include "RcppArmadillo.h"
#include <thread>

// [[Rcpp::depends(RcppArmadillo)]]

double qnor(double p, int lower_tail, int log_p);
double pnor(double x, double mu, double sigma, int lower_tail, int log_p);

void isPD(double value);
void isEAD(double value);
void isLGD(double value);
void isWeight(arma::vec value);

class Counterparty
{
private:
  double pd, ead, lgd, wi, eadxlgd, npd;
  arma::vec weight;
public:
  Counterparty() = default;
  Counterparty(double pd, double ead, double lgd, arma::vec weight);
  ~Counterparty() = default;
  
  double getPD();
  double getNPD();
  double getLGD();
  double getEAD();
  arma::vec getWeights();
  double getWI();
  
  double loss(const arma::mat & Sn);
};


class Portfolio
{
private:
  std::vector<Counterparty> counterparties;
public:
  Portfolio() = default;
  ~Portfolio() = default;
  
  unsigned int getN();
  void addCounterparty(Counterparty c);
  Counterparty getCounterparty(unsigned int i);
  
  double mloss(const arma::mat Vn);
  void ploss(arma::vec * l, const arma::mat * Sn, unsigned int id, unsigned int p);
  arma::vec loss(arma::mat & Sn);
};

#endif