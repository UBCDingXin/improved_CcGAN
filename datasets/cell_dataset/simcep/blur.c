/*
 * =============================================================
 * Blurring in C
 * =============================================================
 * (C) Pekka Ruusuvuori 21.10.2005
 * Modified by Antti Lehmussola 1.2.2006
 * =============================================================
 */

#include "mex.h"
#include "math.h"

void scale_inputs(double *syv, double mi, double ma, int m, int n)
{
    int i,j = 0;
    double maksi = -99999;
    double mini = 99999;
    /* etsitaan minimi ja maksimi */
    for (i = 0; i < n; i++){
        for (j = 0; j < n; j++){
            if (*(syv+(i*m+j)) < mini){
                mini = *(syv+(i*m+j));
            }
            if (*(syv+(i*m+j)) > maksi){
                maksi = *(syv+(i*m+j));
            }
        }
    }
    /* minimia ei ole annettu */
    if (mi == 99999){
        mi = mini;
    }
    /* maksimia ei ole annettu */
    if (mi == -99999){
        ma = maksi;
    }    
    /* skaalaus */
    for (i = 0; i < n; i++){
        for (j = 0; j < m; j++){
            *(syv+(i*m+j)) = *(syv+(i*m+j)) - mini;
            *(syv+(i*m+j)) = (*(syv+(i*m+j)) / (maksi-mini))  * (ma-mi) + mi;
        }
    }
}

void blur(double *C, double *y, double *syv, int win, int m, int n)
{
    int i,j,k,l,count = 0;
    int ci,ci1,ci2,ind1,ind2 = 0;
    int *xx,*yy;
    double *arg,*h;
    double maxh = -99999;
    double myeps, summa, summah, suodatus = 0;
    int w = 0;
    w = (int) floor(win/2);
    myeps = mxGetEps();  /* Matlabin eps */
    
    xx = mxCalloc(win*win,sizeof(int));
    yy = mxCalloc(win*win,sizeof(int));
    arg = mxCalloc(win*win,sizeof(double));
    h = mxCalloc(win*win,sizeof(double));

    for (i = w+1; i < n-(w+1); i++) {
        for (j = w+1; j < m-(w+1); j++) {            

            /* suodin */
            summah = 0;
            ci = 0;
            for (ind1 = -w; ind1 < w+1; ind1++){
                ci2 = 0;
                for (ind2 = -w; ind2 < w+1; ind2++){
                    xx[ci] = ind1;
                    yy[ci] = ind2;
                    arg[ci] = -(xx[ci]*xx[ci] + yy[ci]*yy[ci])/(2 * *(syv+(i*m+j)) * *(syv+(i*m+j)));
                    h[ci] = exp(arg[ci]);
                    if (h[ci] > maxh){
                        maxh = h[ci];
                    }
                    ci2++;
                    ci++;
                }
                ci1++;
            }
            if (ci!=win*win){
                mexPrintf("Error in filter construction!");
            }
            for (ind1 = 0; ind1 < ci; ind1++){  
                /* tuhotaan liian pienet arvot */
                if (h[ind1] < myeps*maxh){
                    h[ind1] = 0;
                }
                summah = summah + h[ind1];
            }
            if (summah!=0){ /* suotimen normalisointi */
                for (ind1 = 0; ind1 < ci; ind1++){
                    h[ind1] = h[ind1]/summah;
                }
            }
            
            summa = 0;
            /* suodatus ikkunalla */
            for (k = -w; k < w+1; k++) {
                for (l = -w; l < w+1; l++) {
                    /*summa = summa + *(y+(i*m+j + k*m+l)) * *(h+(k*win+l + win*w+w));*/
                    /**(C+(i*m+j + k*m+l)) = *(C+(i*m+j + k*m+l))+*(y+(i*m+j + k*m+l)) * *(h+(k*win+l + win*w+w));*/
                    *(C+(i*m+j + k*m+l)) = *(C+(i*m+j + k*m+l))+*(y+(i*m+j + 0)) * *(h+(k*win+l + win*w+w));
                }
            }
            /**(C+(i*m+j)) = summa;*/
        }
    }
}


/* The gateway routine */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *y, *syv, *C;
    double win = 9;
    double mi,ma = 0;
    int mrows,ncols,mrows_syv,ncols_syv;
    int tmp1,tmp2,countti = 0;
    
  /*  Check for proper number of arguments. */
  /* NOTE: You do not need an else statement when using
     mexErrMsgTxt within an if statement. It will never
     get to the else statement if mexErrMsgTxt is executed.
     (mexErrMsgTxt breaks you out of the MEX-file.)
   */
    if (nrhs < 3)
        mexErrMsgTxt("At least two inputs required. See 'help blur'.");
    if (nlhs != 1)
        mexErrMsgTxt("One output required.");
    
    if (nrhs > 2){
  /* Check to make sure the third input argument is a scalar. */
        if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) ||
        mxGetN(prhs[2])*mxGetM(prhs[2]) != 1) {
            mexErrMsgTxt("Input win must be a scalar.");
        }
        
  /* Get the scalar inputs win, mi and ma. */
        win = mxGetScalar(prhs[2]);
    }
    if (nrhs > 3){
        mi = mxGetScalar(prhs[3]);
    } else {
        mi = 99999;
    }
    if (nrhs > 4){
        ma = mxGetScalar(prhs[4]);
    } else {
        ma = -99999;
    }
  /* Create a pointer to the input matrix y. */
    y = mxGetPr(prhs[0]);
  /* Get the dimensions of the matrix input y. */
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);    
  /* Create a pointer to the input matrix syv. */
    syv = mxGetPr(prhs[1]);
  /* Get the dimensions of the matrix input syv. */
    mrows_syv = mxGetM(prhs[1]);
    ncols_syv = mxGetN(prhs[1]);
    if (mrows!=mrows_syv || ncols!=ncols_syv){
        mexErrMsgTxt("The dimensions of input parameters 1 and 2 should match.");
    }    
  /* Set the output pointer to the output matrix. */
    plhs[0] = mxCreateDoubleMatrix(mrows,ncols, mxREAL);
    
  /* Create a C pointer to a copy of the output matrix. */
    C = mxGetPr(plhs[0]);
    
  /* Call the C subroutines. */
    scale_inputs(syv,mi,ma,mrows,ncols);
    blur(C,y,syv,win,mrows,ncols);
}
