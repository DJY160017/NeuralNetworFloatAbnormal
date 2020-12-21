//
// Created by byron on 11/20/19.
//
#include <iostream>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>


using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

int main() {
    Scope root = Scope::NewRootScope();

    auto A = Const(root, {{3.f,  2.f},
                          {-1.f, 0.f}});

    auto b = Const(root, {{3.f, 5.f}});

    auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));

    vector<Tensor> outputs;
    ClientSession session(root);

    TF_CHECK_OK(session.Run({v}, &outputs));
    cout << "tensorflow session run ok!" << endl;
    cout << outputs[0].matrix<float>();
    return 0;
}