#include "credit.h"

void isPD(double value)
{
  if ((value < 0) | (value > 1))
  {
    throw std::invalid_argument("PD must be between 0 and 1");
  }
}

void isEAD(double value)
{
  if (value < 0)
  {
    throw std::invalid_argument("EAD must be positive");
  }
}

void isLGD(double value)
{
  if ((value < 0) | (value > 1))
  {
    throw std::invalid_argument("LGD must be between 0 and 1");
  }
}

void isWeight(arma::vec value)
{
  if (arma::accu(pow(value, 2)) > 1)
  {
    throw std::invalid_argument("Invalid weights");
  }

}

Counterparty::Counterparty(double pd, double ead, double lgd, arma::vec weight): 
  pd(pd), ead(ead), lgd(lgd), weight(weight)
{
  isPD(pd);
  isEAD(ead);
  isLGD(lgd);
  isWeight(weight);
  this->wi = sqrt(1 - arma::accu(pow(weight, 2)));
  this->eadxlgd = lgd * ead;
}

double Counterparty::getPD()
{
  return this->pd;
}

double Counterparty::getLGD()
{
  return this->lgd;
}
  
double Counterparty::getEAD()
{
  return this->ead;
}

arma::vec Counterparty::getWeights()
{
  return this->weight;
}

double Counterparty::getWI()
{
  return this->wi;
}

double Counterparty::loss(const arma::vec & Sn, double & Si)
{
  if (Sn.size() != this->weight.size()) throw std::invalid_argument("Incorrect dimension of Sn");
  
  double CWI = arma::accu(Sn % this->weight) + this->wi * Si;
  return this->pd > CWI ? 0 : this->eadxlgd;
}

int Portfolio::getN()
{
  return this->counterparties.size();
}

void Portfolio::addCounterparty(Counterparty c)
{
  this->counterparties.push_back(c);
}

Counterparty Portfolio::getCounterparty(int i)
{
  if ((i < 0) | (i >= getN())) 
  {
    throw std::invalid_argument("i out of bounds");
  }
  
  return this->counterparties[i];
}

double Portfolio::mloss(const arma::vec & Vn)
{
  double loss(0), d;
  for (auto &&i: this->counterparties)
  {
    d = arma::randn();
    loss += i.loss(Vn, d);
  }
  return loss;
}

void Portfolio::ploss(arma::vec * l, const arma::mat & Sn, int seed, unsigned int id, unsigned int p)
{
  while (id < Sn.n_rows)
  {
    arma::arma_rng::set_seed(seed);
    Sn.row(id).print();
    l->row(id) = mloss(Sn.row(id));
    id += p;
  }
}

arma::vec Portfolio::loss(arma::mat & Sn, int seed)
{
  // unsigned int p = std::thread::hardware_concurrency();
  unsigned int p = 1;
  
  arma::vec l(Sn.n_rows);
  
  std::thread * threads = new std::thread[p];
  
  for (unsigned int i = 0; i < p; i++)
  {
    threads[i] = std::thread(&Portfolio::ploss, this, &l, Sn, seed + i, i, p);
  }
  
  for (unsigned int i = 0; i < p; i++)
  {
    threads[i].join();
  }
  
  return l;
}

RCPP_EXPOSED_CLASS(Counterparty)

RCPP_MODULE(Counterparty) {
  using namespace Rcpp;
  Rcpp::class_<Counterparty>( "Counterparty" )
    .constructor<double, double, double, arma::vec>("Create an object of class bucket")
    .method( "getPD", &Counterparty::getPD)
    .method( "getLGD", &Counterparty::getLGD)
    .method( "getEAD", &Counterparty::getEAD)
    .method( "getWeights", &Counterparty::getWeights)
    .method( "getWI", &Counterparty::getWI)
    .method( "loss", &Counterparty::loss)
  ;
  Rcpp::class_<Portfolio>( "Portfolio" )
    .constructor("Create an empty portfolio")
    .method( "getN", &Portfolio::getN)
    .method( "addCounterparty", &Portfolio::addCounterparty)
    .method( "getCounterparty", &Portfolio::getCounterparty)
    .method( "mloss", &Portfolio::mloss)
    .method( "loss", &Portfolio::loss)
  ;
}


