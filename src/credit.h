#ifndef credit_cpp
#define credit_cpp

#include "RcppArmadillo.h"
#include <thread>

// [[Rcpp::depends(RcppArmadillo)]]

void isPD(double value);
void isEAD(double value);
void isLGD(double value);
void isWeight(arma::vec value);

class Counterparty
{
private:
  double pd, ead, lgd, wi, eadxlgd;
  arma::vec weight;
public:
  Counterparty() = default;
  Counterparty(double pd, double ead, double lgd, arma::vec weight);
  ~Counterparty() = default;
  
  double getPD();
  double getLGD();
  double getEAD();
  arma::vec getWeights();
  double getWI();
  
  double loss(const arma::vec & Sn, double & Si);
};


class Portfolio
{
private:
  std::vector<Counterparty> counterparties;
public:
  Portfolio() = default;
  ~Portfolio() = default;
  
  int getN();
  void addCounterparty(Counterparty c);
  Counterparty getCounterparty(int i);
  
  double mloss(const arma::vec & Vn);
  void ploss(arma::vec * l, const arma::mat & Sn, int seed, unsigned int id, unsigned int p);
  arma::vec loss(arma::mat & Sn, int seed = 12345);
};

#endif