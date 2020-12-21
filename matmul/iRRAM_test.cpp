//
// Created by byron on 11/20/19.
//
#include <iostream>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <iRRAM/lib.h>
#include <vector>
#include <typeinfo>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

extern int iRRAM::MAXiterationnum = 0;
extern bool iRRAM::enableReiterate = false;
extern bool iRRAM::alwaysenableReiterate = false;

void compute();
int iRRAM_compute(const int &arg){
    compute();
    return 0;
}

int main(int argc, char **argv){
    iRRAM::cout.real_w = 50;

    // cout << "test tensor init and calculate start" << endl;
    // Scope root = Scope::NewRootScope();
    // auto A = Const(root, {{3.f, 2.f},
    //                       {-1.f, 0.f}});

    // auto b = Const(root, {{3.f, 5.f}});

    // auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));

    // vector<Tensor> outputs;

    // ClientSession session(root);

    // TF_CHECK_OK(session.Run({v}, &outputs));
    // auto print_content = outputs[0].matrix<float>();
    // std::cout<<print_content<<std::endl;
    // cout << *print_content.data() << endl;
    // cout << "tensorflow session run ok!" << endl;
    // cout << "test tensor init and calculate end" << endl;
    // cout << "======================================================================================" << endl;
    
    cout<<"test iRRAM and tensor init start"<<endl;
    iRRAM_initialize(argc,argv);

    int result = iRRAM::iRRAM_exec(iRRAM_compute, 0);
    cout<<"test iRRAM and tensor init end and exitcode is: "<<result<<endl;
    return result;
}

void compute()
{

    Scope root1 = Scope::NewRootScope();

    cout << "test 1 judge the type" << endl;
    iRRAM::REAL real_x_1 = 3.5;
    iRRAM::REAL real_y_1 = 2;
    Tensor x_1(real_x_1);
    Tensor y_1(real_y_1);
    cout << "real type:" << x_1.dtype() << endl;
    cout << "expect type:" << DT_REAL << endl;

    cout << "test 2" << endl;
    iRRAM::REAL real_x_2 = -1;
    iRRAM::REAL real_y_2 = 0;
    Tensor x_2(real_x_2);
    Tensor y_2(real_y_2);

    cout << "test 3" << endl;
    iRRAM::REAL real_x_3 = 3;
    iRRAM::REAL real_y_3 = 5;
    Tensor x_3(real_x_3);
    Tensor y_3(real_y_3);

    cout << "test 4 construct a" << endl;
    auto a_m = Const(root1, {{real_x_1, real_y_1},
                             {real_x_2, real_y_2}});
    cout << "test 4 construct b" << endl;
    auto b_m = Const(root1, {{real_x_3, real_y_3}});

    cout << "test 5" << endl;

    auto result = MatMul(root1.WithOpName("v1"), a_m, b_m, MatMul::TransposeB(true));

    std::cout<<"test 6 result type is: "<<typeid(result).name()<<std::endl;

    vector<Tensor> outputs1;
    ClientSession session1(root1);

    cout << "test 6" << endl;
    TF_CHECK_OK(session1.Run({result}, &outputs1));

    cout << "test 7" << endl;
    Tensor last_result = outputs1[0];
    auto print_content = last_result.matrix<iRRAM::REAL>();
    iRRAM::REAL *real_value = print_content.data();
    iRRAM::cout << (*real_value) << endl;
    cout << "tensorflow session1 run ok!" << endl;
}