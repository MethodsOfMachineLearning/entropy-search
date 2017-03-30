
#include "mex.h"

#include <Eigen/Dense>
#include <cmath>
#include <limits>

#define FIXNAN() unaryExpr(std::ptr_fun(fixNaN))


double fixNaN(double x) {

    return std::isnan(x) ? 0 : x;
}

double fixLogInf(double x) {

    return std::isinf(x) ? -500 : x;
}

void printMatrix(char *s, Eigen::MatrixXd M) {

  mexPrintf("Matrix: %s\n", s);

  for (int i = 0; i < M.rows(); i++) {

    for (int j = 0; j < M.cols(); j++) {
      mexPrintf("\t%f", M(i, j));
    }
    mexPrintf("\n");
  }
  mexPrintf("\n\n");
}

Eigen::VectorXd logsumexp(Eigen::MatrixXd x) {

  Eigen::VectorXd y = x.colwise().maxCoeff();

  Eigen::VectorXd s = (x.rowwise() - y.transpose()).array().exp().colwise().sum().log();

  return y + s;
}

double logdet(Eigen::MatrixXd M) {

  Eigen::LLT<Eigen::MatrixXd> llt(M);

  // D = 2 * sum(log(diag(chol(M))));

  Eigen::VectorXd res = ((Eigen::MatrixXd) llt.matrixU()) // Calc chol())
          .diagonal().array().log() // Log of diagonal
          .colwise().sum(); // Calc sum of logs

  return 2 * res[0];
}

static inline double max(double a, double b) {
  return a < b ? b : a;
}

static inline double log_relative_Gauss(double z, double &e, int &exit_flag) {

  double logphi, logPhi;

  if (z < -6) {

    e = 1;
    logPhi = -1.0e12;
    exit_flag = -1;

  } else if (z > 6) {

    e = 0;
    logPhi = 0;
    exit_flag = 1;

  } else {

    logphi = -0.5 * (z * z + log(M_PI * 2)); // Const function call gets optimized away
    logPhi = log(0.5 * erfc(-z * M_SQRT1_2));
    e = exp(logphi - logPhi);
    exit_flag = 0;
  }
  return logPhi;
}

static void lt_factor(int s, int l, Eigen::VectorXd M, Eigen::MatrixXd V, double mp, double p, double gam,
        Eigen::VectorXd &Mnew, Eigen::MatrixXd &Vnew, double &pnew, double &mpnew, double &logS, double &d) {

  // rank 1 projected cavity parameters
  Eigen::VectorXd Vc = (V.col(l) - V.col(s)) * M_SQRT1_2;
  double cVc = (V(l, l) - 2 * V(s, l) + V(s, s)) / 2;
  double cM = (M(l) - M(s)) * M_SQRT1_2;

  double cVnic = max(0, cVc / (1 - p * cVc));

  double cmni = cM + cVnic * (p * cM - mp);

  // rank 1 calculation: step factor
  double z = cmni / sqrt(cVnic);

  double e;
  int exit_flag;
  double lP = log_relative_Gauss(z, e, exit_flag);

  double alpha, beta, r, dp, dmp;

  switch (exit_flag) {

    case 0:

      alpha = e / sqrt(cVnic);
      beta = alpha * (alpha * cVnic + cmni);
      r = beta / (1 - beta);

      // new message
      pnew = r / cVnic;
      mpnew = r * (alpha + cmni / cVnic) + alpha;

      // update terms
      dp = max(-p + DBL_EPSILON, gam * (pnew - p));
      dmp = max(-mp + DBL_EPSILON, gam * (mpnew - mp));
      d = max(dmp, dp); // for convergence measures

      pnew = p + dp;
      mpnew = mp + dmp;

      // project out to marginal
      Vnew = V - dp / (1 + dp * cVc) * (Vc * Vc.transpose());
      Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;

      // normalization constant
      //logS  = lP - 0.5 * (log(beta) - log(pnew)) + (alpha * alpha) / (2*beta);

      // there is a problem here, when z is very large
      logS = lP - 0.5 * (log(beta) - log(pnew) - log(cVnic)) + (alpha * alpha) / (2 * beta) * cVnic;

      break;

    case -1: // impossible combination

      d = NAN;

      //Mnew = 0;
      //Vnew = 0;

      pnew = 0;
      mpnew = 0;
      logS = -INFINITY;
      break;

    case 1: // uninformative message

      pnew = 0;
      mpnew = 0;

      // update terms
      dp = -p; // at worst, remove message
      dmp = -mp;
      d = max(dmp, dp); // for convergence measures

      // project out to marginal
      Vnew = V - dp / (1 + dp * cVc) * (Vc * Vc.transpose());
      Mnew = M + (dmp - cM * dp) / (1 + dp * cVc) * Vc;

      logS = 0;
      break;
  }
}

double min_factor(Eigen::VectorXd Mu, Eigen::MatrixXd Sigma, int k, double gam,
        Eigen::VectorXd &dlogZdMu, Eigen::VectorXd &dlogZdSigma, Eigen::MatrixXd &dlogZdMudMu) {

  int D = Mu.size();

  double logZ;

  // messages (in natural parameters)
  Eigen::VectorXd logS = Eigen::VectorXd::Zero(D - 1); // normalization constant (determines zeroth moment)
  Eigen::VectorXd MP = Eigen::VectorXd::Zero(D - 1); // mean times precision (determines first moment)
  Eigen::VectorXd P = Eigen::VectorXd::Zero(D - 1); // precision (determines second moment)  

  // TODO: check if copy is really necessary here
  // marginal:
  Eigen::VectorXd M(Mu);
  Eigen::MatrixXd V(Sigma);

  double mpm;
  double s;
  double rSr;
  double dts;

  //Eigen::VectorXd dMdMu;
  //Eigen::VectorXd dMdSigma;
  //Eigen::VectorXd dVdSigma;
  Eigen::MatrixXd _dlogZdSigma;

  Eigen::MatrixXd R;
  Eigen::VectorXd r;

  Eigen::MatrixXd IRSR;
  Eigen::MatrixXd A;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b;
  Eigen::VectorXd Ab;

  Eigen::VectorXd btA;

  Eigen::MatrixXd C;

  double pnew;
  double mpnew;

  double Diff = 0, diff = 0;

  int l, count = 0;

  // mvmin = Eigen::VectorXd(2);

  while (true) {

    count++;

    Diff = 0;

    for (int i = 0; i < D - 1; i++) {

      if (i < k)
        l = i;
      else
        l = i + 1;

      lt_factor(k, l, M, V, MP[i], P[i], gam, // IN
              M, V, pnew, mpnew, logS[i], diff); // OUT

      // Write back vector elements
      P[i] = pnew;
      MP[i] = mpnew;

      if (std::isnan(diff))
        goto done; // found impossible combination

      Diff = Diff + std::abs(diff);
    }

    if (count > 50) {
      mexPrintf("EP iteration ran over iteration limit. Stopped.\n");
      goto done;
    }
    if (Diff < 1.0e-3) {
      goto done;
    }
  }

done:

  if (std::isnan(diff)) {

    logZ = -INFINITY;
    dlogZdMu = Eigen::VectorXd::Zero(D);
    dlogZdSigma = Eigen::VectorXd::Zero(0.5 * (D * (D + 1)));
    dlogZdMudMu = Eigen::MatrixXd::Zero(D, D);
    //mvmin << Mu(k), Sigma(k, k);
    //dMdMu = Eigen::VectorXd::Zero(D);
    //dMdSigma = Eigen::VectorXd::Zero(0.5 * (D * (D + 1)));
    //dVdSigma = Eigen::VectorXd::Zero(0.5 * (D * (D + 1)));

  } else {

    // evaluate log Z:

    // C = eye(D) ./ sqrt(2); C(k,:) = -1/sqrt(2); C(:,k) = [];
    C = Eigen::MatrixXd::Zero(D, D - 1);
    for (int i = 0; i < D - 1; i++) {

      C(i + (i >= k), i) = M_SQRT1_2;
      C(k, i) = -M_SQRT1_2;
    }

    R = C.array().rowwise() * P.transpose().array().sqrt();
    r = (C.array().rowwise() * MP.transpose().array()).rowwise().sum();
    mpm = (MP.array() * MP.array() / P.array()).FIXNAN().sum();
    s = logS.sum();

    IRSR = R.transpose() * Sigma * R;
    IRSR.diagonal().array() += 1; // Add eye()

    rSr = r.dot(Sigma * r);

    A_ = R * IRSR.llt().solve(R.transpose());
    A = 0.5 * (A_.transpose() + A_); // ensure symmetry.

    b = (Mu + Sigma * r);
    Ab = A * b;

    dts = logdet(IRSR);
    logZ = 0.5 * (rSr - b.dot(Ab) - dts) + Mu.dot(r) + s - 0.5 * mpm;

    if (true /*TODO: needs derivative? */) {

      dlogZdSigma = Eigen::VectorXd(0.5 * (D * (D + 1)));

      btA = b.transpose() * A;

      dlogZdMu = r - Ab;
      dlogZdMudMu = -A;

      _dlogZdSigma = -A - 2 * r * Ab.transpose() + r * r.transpose() + btA * Ab.transpose();


      Eigen::MatrixXd diag = _dlogZdSigma.diagonal().asDiagonal();

      _dlogZdSigma = 0.5 * (_dlogZdSigma + _dlogZdSigma.transpose() - diag);

      // dlogZdSigma = dlogZdSigma(logical(triu(ones(D,D))));
      for (int x = 0, i = 0; x < D; x++) {

        for (int y = 0; y <= x; y++) {
          dlogZdSigma[i++] = _dlogZdSigma(y, x);
        }
      }
    }
  }

  return logZ;
}

void joint_min(Eigen::VectorXd Mu, Eigen::MatrixXd Sigma,
        Eigen::VectorXd &logP, Eigen::MatrixXd &dlogPdMu, Eigen::MatrixXd &dlogPdSigma, Eigen::MatrixXd **dlogPdMudMu) {

  Eigen::VectorXd dlPdM;
  Eigen::VectorXd dlPdS;
  Eigen::MatrixXd dlPdMdM;

  double gam = 1;
  int D = Mu.size();

  Eigen::MatrixXd gg = Eigen::MatrixXd(D, D);

  logP = Eigen::VectorXd(D);

  dlogPdMu = Eigen::MatrixXd(D, D);
  dlogPdSigma = Eigen::MatrixXd(D, D * (D + 1) / 2);
  *dlogPdMudMu = new Eigen::MatrixXd[D]; // Create an array of matrizes

  for (int k = 0; k < D; k++) {

#ifdef DEBUG_PRINTF
    if (k % 10 == 0)
      DEBUG_PRINTF('#');
#endif
    
    logP(k) = min_factor(Mu, Sigma, k, gam, // IN
            dlPdM, dlPdS, dlPdMdM); // OUT

    dlogPdMu.row(k) = dlPdM;
    dlogPdSigma.row(k) = dlPdS;

    (*dlogPdMudMu)[k] = dlPdMdM;
  }

  // Sanity check for INF values
  logP = logP.unaryExpr(std::ptr_fun(fixLogInf));

  // re-normalize at the end, to smooth out numerical imbalances:
  double Z = logP.array().exp().sum();

  Eigen::VectorXd Zm = (dlogPdMu.array().colwise() * logP.array().exp()).colwise().sum() /Z;
  Eigen::VectorXd Zs = (dlogPdSigma.array().colwise() * logP.array().exp()).colwise().sum() /Z;

  Eigen::MatrixXd Zij = Zm * Zm.transpose();

  for (int i = 0; i < D; i++) {

    for (int j = i; j < D; j++) {

      Eigen::MatrixXd Mj = (*dlogPdMudMu)[j];

      for (int k = 0; k < D; k++) {
        gg(i, j) -= (dlogPdMu(k, i) * dlogPdMu(k, j) + Mj(k, i)) * exp(logP(k));
      }
      gg(j, i) = // Hesse Matrix is symmetric
              gg(i, j) = gg(i, j) / Z + Zij(i, j);
    }
  }

  for (int i = 0; i < D; i++) {
    (*dlogPdMudMu)[i].array() *= gg.array();
  }

  dlogPdMu = dlogPdMu.array().rowwise() - Zm.transpose().array();
  dlogPdSigma = dlogPdSigma.array().rowwise() - Zs.transpose().array();
  
  logP = logP.array() - logsumexp(logP)(0, 0);
}

static void copyMatrix(Eigen::MatrixXd from, mxArray **out) {

  int r = from.rows();
  int c = from.cols();

  *out = mxCreateDoubleMatrix(r, c, mxREAL);
  double *write = mxGetPr(*out);
  for (int i = 0; i < r; i++) {

    for (int j = 0; j < c; j++) {
      write[j * r + i] = from(i, j);
    }
  }
}

static void copyCube(Eigen::MatrixXd *from, int elms, mxArray **out) {

  // They're all of the same dimension
  int r = from->rows();
  int c = from->cols();

  mwSize dims[3];
  dims[0] = r;
  dims[1] = c;
  dims[2] = elms;

  *out = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  double *write = mxGetPr(*out);
  for (int k = 0; k < elms; k++) {

    for (int i = 0; i < r; i++) {

      for (int j = 0; j < c; j++) {
        write[k * r * c + i * c + j] = from[k](i, j);
      }
    }
  }
}

void mexFunction(int nlhs, mxArray * plhs[],
        int nrhs, const mxArray * prhs[]) {

  if (nrhs != 2)
    mexErrMsgIdAndTxt("MATLAB:xtimesy:invalidNumInputs",
          "Two inputs required.");

  if (nlhs != 4)
    mexErrMsgIdAndTxt("MATLAB:xtimesy:invalidNumOutputs",
          "Four outputs required.");

  if (mxGetM(prhs[0]) != 1 && mxGetN(prhs[0]) != 1) {
    mexErrMsgIdAndTxt("MATLAB:xtimesy:invalidNumOutputs",
            "Vector for param 1 required");
  }

  // Input vars
  Eigen::Map<Eigen::VectorXd> Mu(mxGetPr(prhs[0]), mxGetM(prhs[0]) == 1 ? mxGetN(prhs[0]) : mxGetM(prhs[0]));
  Eigen::Map<Eigen::MatrixXd> Sigma(mxGetPr(prhs[1]), mxGetM(prhs[1]), mxGetN(prhs[1]));

  // Output vars
  Eigen::VectorXd logP;
  Eigen::MatrixXd dlogPdMu;
  Eigen::MatrixXd dlogPdSigma;
  Eigen::MatrixXd *dlogPdMudMu;

  // Do the heavy work
  joint_min(Mu, Sigma,
          logP, dlogPdMu, dlogPdSigma, &dlogPdMudMu);

  // Output results
  copyMatrix(logP, &(plhs[0]));
  copyMatrix(dlogPdMu, &(plhs[1]));
  copyMatrix(dlogPdSigma, &(plhs[2]));
  copyMatrix(dlogPdSigma, &(plhs[3]));
  copyCube(dlogPdMudMu, Mu.size(), &(plhs[3]));

  // Endpoint, delete array
  delete[] dlogPdMudMu;
}
