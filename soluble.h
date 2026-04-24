#include "redistance2.h"
#include "diffusion.h"
#include "tracer.h"

#define EPSILON ((L0/(1 << grid->maxdepth))*0.75)
//#define EPSILON 0.08//((L0/(1 << grid->maxdepth))*0.25)

bool advect_diff_phase_field = 0;
int reinit_skip_steps = 100;
bool surfactant = 1;
int counter = 0;

scalar ci[], cb[], gamma2[], d2[], pfield[];
scalar * tracers = {ci, cb, d2, pfield};

double zeta = 1. [*];
double Di0 = 0.01 [*], Db0 = 0.01 [*];
double ra = 1. [*], rd = 1. [*];
double c_hat_inf = 1. [*];
double varepsilon = 1.e-12;
double sharpening_coefficient = 2.;
double inverse_pfield = 1; //inverting the field according to problem

struct HDiffusion {
    face vector D;
    face vector beta;
    scalar lambda;
};

static void h_relax(scalar *al, scalar *bl, int l, void *data) {
    scalar a = al[0], b = bl[0];
    struct HDiffusion *p = (struct HDiffusion *) data;
    face vector D = p->D, beta = p->beta;
    scalar lambda = p->lambda;

    scalar c = a;
    foreach_level_or_leaf(l) {
        double n = -sq(Delta)*b[], d = -lambda[]*sq(Delta);

        foreach_dimension() {
            n += D.x[1]*a[1] + D.x[]*a[-1] + 0.5*Delta*(beta.x[1]*a[1] - beta.x[]*a[-1]);
            d += D.x[1] + D.x[] - 0.5*Delta*(beta.x[1] - beta.x[]);
        }
        c[] = n / d;
    }
}

static double h_residual(scalar *al, scalar *bl, scalar *resl, void *data) {
    scalar a = al[0], b = bl[0], res = resl[0];
    struct HDiffusion *p = (struct HDiffusion *) data;
    face vector D = p->D, beta = p->beta;
    scalar lambda = p->lambda;
    double maxres = 0.;

#if TREE
    face vector g[];
    foreach_face() {
        g.x[] = D.x[]*face_gradient_x(a, 0) + beta.x[]*face_value(a, 0);
    }
    foreach(reduction(max:maxres)) {
        res[] = b[] - lambda[]*a[];
        // res[] = b[] + cm[]/dt*a[];

        foreach_dimension() {
            res[] -= (g.x[1] - g.x[]) / Delta;
        }
        if (fabs(res[]) > maxres) maxres = fabs(res[]);
    }
#else
    foreach(reduction(max:maxres)) {
        res[] = b[] - lambda[]*a[];
        foreach_dimension() {
            res[] -= (D.x[1]*face_gradient_x(a, 1) - D.x[0]*face_gradient_x(a, 0) + beta.x[1]*face_value(a, 1) - beta.x[0]*face_value(a, 0))/Delta;
        }
        if (fabs(res[] > maxres)) maxres = fabs(res[]);
    }
#endif

    return maxres;
}

event stability(i++) {
    double maxvel = 1.e-6 [*];
    double Dmax = max(Di0, Db0);
    
    if (surfactant) {
        foreach_face(reduction(max:maxvel)) {
            if (u.x[] != 0.) {
                double deltas = (pfield[]*(1. - pfield[])) / EPSILON;
                if (deltas > 0.01) {
                    if (fabs(u.x[]) > maxvel) maxvel = fabs(u.x[]);
                }
            }
        }

        double deltaxmin = 1. [*];
        deltaxmin = L0 / (1 << grid->maxdepth);
        zeta = 1.1*maxvel;
        double dt = dtmax;
        
        if (advect_diff_phase_field) {
            dt = 0.75*sq(deltaxmin) / zeta / EPSILON;
            if (dt < dtmax) dtmax = dt;
        }

        dt = 0.75*sq(deltaxmin) / 4. / Dmax;
#if dimension == 3
        dt = 0.75*sq(deltaxmin) / 6. / Dmax;
#endif
    }
}

double clamp2(double a) {
    if (a < 1.e-6) return 0;
    else if (a > 1. - 1.e-6) return 1.;
    else return a;
}

event properties2(i++) {
    if (reinit_skip_steps > 1) advect_diff_phase_field = 1;
    else advect_diff_phase_field = 0;

    if (counter >= 0 && counter%reinit_skip_steps == 0) {
        scalar d2temp[];
        double deltamin = L0 / (1 << grid->maxdepth);
        foreach() {
            d2temp[] = (2.*f[] - 1.)*deltamin*0.75;
        }
#if TREE
        restriction({d2temp});
#endif
        redistance(d2temp, imax = 0.5*(1 << grid->maxdepth));
        
        double d2max = statsf(d2temp).max;
        double d2min = statsf(d2temp).min;
        bool signed_distance_faulty = 0;

        if (d2max > 6. || d2min < -6.) {
            signed_distance_faulty = 1;
            counter = counter - 2;
        }

        if (!signed_distance_faulty) {
            foreach() {
                d2[] = inverse_pfield*d2temp[];
                pfield[] = 0.5*(1. - tanh((d2[] / 2. / EPSILON)));
                pfield[] = clamp2(pfield[]);
            }
            boundary({pfield});
        }
    }
    counter ++;
}

event tracer_diffusion(i++) {
    if (surfactant) {

        scalar rhs[], psi[], r[], lambda[];
        face vector cflux[], geta[], beta[], D[];
        face vector betai[], betab[], Di[], Db[];
        scalar ri[], rb[], lambdai[], lambdab[];

        if (advect_diff_phase_field) {
            foreach() {
                pfield[] = clamp2(pfield[]);
                psi[] = EPSILON*log((pfield[] + varepsilon) / (1. - pfield[] + varepsilon));
                lambda[] = - cm[] / dt;
            }
            boundary({psi});

            foreach_face() {
                cflux.x[] = 0.;
                double psif = (psi[] + psi[-1])/2.;
                double gradpsi = (psi[] - psi[-1])/Delta;

                if (fabs(gradpsi) > varepsilon) {

#if dimension == 2
                    double psiup = 0.25*(psi[] + psi[-1] + psi[0,1] + psi[-1,1 ]);
                    double psidown = 0.25*(psi[] + psi[-1] + psi[0,-1] + psi[-1,- 1]);
                    double mag_grad_psi = sqrt(sq(psi[] - psi[-1]) + sq(psiup - psidown)) / Delta;
#endif

#if dimension == 3
                    double psiup = 0.25*(psi[0, 0, 0] + psi[-1, 0, 0] + psi[0, 1, 0] + psi[-1, 1, 0]);
                    double psidown = 0.25*(psi[0, 0, 0] + psi[-1, 0, 0] + psi[0, -1, 0] + psi[-1, -1, 0]);
                    double psifront = 0.25*(psi[0, 0, 0] + psi[-1, 0, 0] + psi[0, 0, 1] + psi[-1, 0, 1]);
                    double psiback = 0.25*(psi[0, 0, 0] + psi[-1, 0, 0] + psi[0, 0, -1] + psi[-1, 0, -1]);
                    double mag_grad_psi = sqrt(sq(psi[0, 0, 0] - psi[-1, 0, 0]) + sq(psiup - psidown) + sq(psifront - psiback)) / Delta;
#endif

                    cflux.x[] = fm.x[]*(1. - sq(tanh(psif/2./EPSILON)))*gradpsi/mag_grad_psi;
                }

                D.x[] = fm.x[]*zeta*EPSILON;
                beta.x[] = 0.;
            }

            foreach() {
#if dimension == 2
                rhs[] = - 0.25*zeta*(cflux.x[1] - cflux.x[] + (cflux.y[0, 1] - cflux.y[])/cm[]) / Delta*cm[];
                r[] = - cm[]*pfield[]/dt + 0.25*zeta*(cflux.x[1] - cflux.x[] + (cflux.y[0, 1] - cflux.y[])) / Delta;
#endif

#if dimension == 3
                rhs[] = - 0.25*zeta*(cflux.x[1] - cflux.x[] + cflux.y[0, 1] - cflux.y[] + cflux.z[0, 0, 1] - cflux.z[]) / Delta*cm[];
                r[] = - cm[]*pfield[]/dt + 0.25*zeta*(cflux.x[1, 0, 0] - cflux.x[0, 0, 0] + cflux.y[0, 1, 0] - cflux.y[0, 0, 0] + cflux.z[0, 0, 1] - cflux.z[0, 0, 0]) / Delta;
#endif
            }

            restriction({D, beta, cm, lambda});
            struct HDiffusion q1;
            q1.D = D;
            q1.beta = beta;
            q1.lambda = lambda;

            mg_solve({pfield}, {r}, h_residual, h_relax, &q1);
        }

        foreach() {
            pfield[] = clamp2(pfield[]);
            double deltas = (pfield[]*(1. - pfield[]))/EPSILON;
            double phi = max(pfield[], 1.e-6);
            psi[] = EPSILON*log((clamp(pfield[],0.,1.) + varepsilon) / (1. - clamp(pfield[],0.,1.) + varepsilon));

#if dimension == 2
            double phiup   = 0.25*(pfield[]   + pfield[-1] + pfield[0,1] + pfield[-1,1]);
            double phidown = 0.25*(pfield[]   + pfield[-1] + pfield[0,-1]+ pfield[-1,-1]);
            double mag_grad_phi = sqrt(sq(pfield[] - pfield[-1]) + sq(phiup - phidown)) / Delta;
#endif

#if dimension == 3
            double phiup    = 0.25*(pfield[0,0,1] + pfield[-1,0,1] + pfield[0,1,1] + pfield[-1,1,1]);
            double phidown  = 0.25*(pfield[0,0,-1]+ pfield[-1,0,-1] + pfield[0,1,-1]+ pfield[-1,1,-1]);
            double phifront = 0.25*(pfield[0,0,0] + pfield[-1,0,0] + pfield[0,0,1] + pfield[-1,0,1]);
            double phiback  = 0.25*(pfield[0,0,0] + pfield[-1,0,0] + pfield[0,0,-1]+ pfield[-1,0,-1]);
            double mag_grad_phi = sqrt(sq(pfield[] - pfield[-1]) + sq(phiup - phidown) + sq(phifront - phiback)) / Delta;
#endif

            //lambdai[] = -cm[]*(1./dt); // to simulate only adsorption
            lambdai[] = -cm[]*(1./dt + ra*cb[]/phi + rd);

            //lambdab[] = -cm[]*(1./dt + ra*c_hat_inf*deltas/phi); // to simulate only adsorption
            lambdab[] = -cm[]*(1./dt + ra*(c_hat_inf*deltas - ci[])/phi);

            //ri[] = -cm[]*(ci[]/dt + ra*cb[]*c_hat_inf*deltas/phi); // to simulate only adsorption
            ri[] = -cm[]*(ci[]/dt + ra*cb[]*c_hat_inf*deltas/phi);
 
            rb[] = -cm[]*(cb[]/dt + rd*ci[]);
        }

        foreach_face() {
            double phif = (pfield[] + pfield[-1])/2.;
            Di.x[] = fm.x[]*Di0;
            Db.x[] = fm.x[]*Db0;
            double gradpsi = (psi[] - psi[-1])/Delta;
            betai.x[] = betab.x[] = 0.;

            if (fabs(gradpsi) > varepsilon) {
#if dimension == 2
                double psiup   = 0.5*(psi[0,1] + psi[-1,1]);
                double psidown = 0.5*(psi[0,-1] + psi[-1,-1]);
                double mag_grad_psi = sqrt(sq(psi[] - psi[-1]) + sq(psiup - psidown)/4.)/Delta;
#endif

#if dimension == 3
                double psiup    = 0.5*(psi[0,1,0] + psi[-1,1,0]);
                double psidown  = 0.5*(psi[0,-1,0] + psi[-1,-1,0]);
                double psifront = 0.5*(psi[0,0,1] + psi[-1,0,1]);
                double psiback  = 0.5*(psi[0,0,-1] + psi[-1,0,-1]);
                double mag_grad_psi = sqrt(sq(psi[] - psi[-1]) + sq(psiup - psidown)/4. + sq(psifront - psiback)/4.)/Delta;
#endif

                betai.x[] = -Di.x[]*sharpening_coefficient*(0.5 - phif)/EPSILON*gradpsi/mag_grad_psi;
                betab.x[] = -Db.x[]*(1. - phif)/EPSILON*gradpsi/mag_grad_psi;
            }     
        }
        
        restriction({Di, Db, betai, betab, lambdai, lambdab, cm, ri, rb});

        struct HDiffusion qi;
        qi.D      = Di;
        qi.beta   = betai;
        qi.lambda = lambdai;

        mg_solve({ci}, {ri}, h_residual, h_relax, &qi);

        struct HDiffusion qb;
        qb.D      = Db;
        qb.beta   = betab;
        qb.lambda = lambdab;
        
        mg_solve({cb}, {rb}, h_residual, h_relax, &qb);
    }  
}