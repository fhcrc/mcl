#include <stdio.h>
#include "util/rand.h"
#include "src/impala/app.h"
#include "src/mcl/alg.h"
#include "caml/mlvalues.h"
#include "caml/memory.h"
#include "caml/alloc.h"
#include "caml/custom.h"

static int caml_mcl_initialized = 0;

void caml_mcl_initialize(void)
{
    if (caml_mcl_initialized)
        return;

    srandom(mcxSeed(315));
    mclx_app_init(NULL);
    caml_mcl_initialized = 1;
}

CAMLprim value caml_mcl(value arr)
{
    CAMLparam1(arr);
    int i, cols = Wosize_val(arr);
    mclv *domc = mclvCanonical(NULL, cols, 1.0);
    mclv *domr = mclvCanonical(NULL, cols, 1.0);
    mclx *res_mat, *mx = mclxAllocZero(domc, domr);
    mclAlgParam *mlp;
    value res;

    for (i = 0; i < cols; ++i) {
        value col = Field(arr, i);
        int j, rows = Wosize_val(col);
        mclv *col_vec = &mx->cols[i];
        if (!cols)
            continue;

        mclvResize(col_vec, rows);
        for (j = 0; j < rows; ++j) {
            value t = Field(col, j);
            col_vec->ivps[j].idx = Int_val(Field(t, 0));
            col_vec->ivps[j].val = Double_val(Field(t, 1));
        }
    }


    mclAlgInterface(&mlp, NULL, 0, NULL, mx, 0);
    mclAlgorithm(mlp);

    res_mat = mlp->cl_result;
    cols = res_mat->dom_cols->n_ivps;
    res = caml_alloc(cols, 0);
    for (i = 0; i < cols; ++i) {
        mclv *col_vec = &res_mat->cols[i];
        int j, rows = col_vec->n_ivps;
        value row = caml_alloc(rows, 0);
        for (j = 0; j < rows; ++j) {
            Store_field(row, j, Val_int(col_vec->ivps[j].idx));
        }
        Store_field(res, i, row);
    }

    mclAlgParamFree(&mlp, TRUE);

    CAMLreturn(res);
}
