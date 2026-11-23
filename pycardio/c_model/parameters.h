struct parameters {
  double eiAd; // slow variable for early-i
  double rs1; // noise to early-i
  
  double hr;
  double rsa1;
  double rsa2;

  double po;
  double taup;
  double taul;
  double taupp;
  
  double baro;
  double sym;
  double db;
  double sb;

  // drives
  double d1;
  double d2;
  double d3;
  double d4;
  double d5;

};

const double st2 = 0.; // stimulus to early-I
const double st4 = 0.; // stimulus to post-I

const double v12mp = -40.; // pre-i/i persistent sodium vinf
const double kmp = -6.; // pre-i/i persistent sodium decay constant
const double EL = -60.; // leak reversal potential
const double C = 20.; // membrane capacitance
const double gL = 2.8; // leak conductance
const double v12hp = -55.; // pre-i/i nap inactivation vinf
const double khp = 10.; // pre-i/i nap inactivation decay constant
const double tauhp = 4000.; // pre-i/i nap inactivation time constant
const double gNaP = 5.; // pre-i/i persistent sodium conductance
const double ENa = 50.; // pre-i/i sodium reversal potential
const double EK=-85.;
const double Ee=0.;
const double Ei=-75.;
const double v12n=-30.;
const double kn=-4.;
const double gKdr=5.;
const double gAD=10.;

const double tad = 2000; // activation time constant
const double tad_ei = 2000; // early-i time constant

const double gE=10.;
const double gI=60.;

    
const double hb = 0.2;
const double hb2 = 0.;


// connectivity
const double a12 = .5;
const double b32 = .05;
const double b42 = .42;

const double b23 = .42;
const double b43 = .2;

const double b24 = .22;
const double b34 = .0;

const double a15 = .5;
const double b25 = .3;
const double b35 = .7;
const double b45 = .7;

const double b41 = .1;
const double b31 = 0.;
