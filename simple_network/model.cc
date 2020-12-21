#include <iostream>
#include <typeinfo>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/cc/framework/gradients.h>

#include "data_helper.h"

using namespace tensorflow;

extern int iRRAM::MAXiterationnum = 0;
extern bool iRRAM::enableReiterate = false;
extern bool iRRAM::alwaysenableReiterate = false;

void compute();
void original_compute();
int iRRAM_compute(const int &arg){
    compute();
    return 0;
}

int main(int argc, char **argv){
    iRRAM::cout.real_w = 50;
    //原版本dnn
    // original_compute();
    //iRRAM版本dnn
    iRRAM_initialize(argc,argv);
    int result = iRRAM::iRRAM_exec(iRRAM_compute, 0);
    std::cout<<"iRRAM exec exitcode is: "<<result<<std::endl;
}

void original_compute(){
    DataHelper<float> data_helper("/home/byron/Documents/workspace/NeuralNetworFloatAbnormal/simple_network/data/", "normalized_car_features.csv");
    Tensor x_data(DataTypeToEnum<float>::v(),
                  TensorShape{static_cast<int>(data_helper.x().size()) / 3, 3});
    copy_n(data_helper.x().begin(), data_helper.x().size(),
           x_data.flat<float>().data());

    Tensor y_data(DataTypeToEnum<float>::v(),
                  TensorShape{static_cast<int>(data_helper.y().size()), 1});
    copy_n(data_helper.y().begin(), data_helper.y().size(),
           y_data.flat<float>().data());

    Scope scope = Scope::NewRootScope();
    std::cout<<"simple_networl test1"<<std::endl;
    auto x = ops::Placeholder(scope, DT_FLOAT);
    auto y = ops::Placeholder(scope, DT_FLOAT);
    std::cout<<"simple_networl test2"<<std::endl;
    // weights init
    auto w1 = ops::Variable(scope, {3, 3}, DT_FLOAT);
    auto assign_w1 = ops::Assign(scope, w1, ops::RandomNormal(scope, {3, 3}, DT_FLOAT));
    std::cout<<"simple_networl test3"<<std::endl;

    auto w2 = ops::Variable(scope, {3, 2}, DT_FLOAT);
    auto assign_w2 = ops::Assign(scope, w2, ops::RandomNormal(scope, {3, 2}, DT_FLOAT));
    std::cout<<"simple_networl test4"<<std::endl;

    auto w3 = ops::Variable(scope, {2, 1}, DT_FLOAT);
    auto assign_w3 = ops::Assign(scope, w3, ops::RandomNormal(scope, {2, 1}, DT_FLOAT));
    std::cout<<"simple_networl tes5"<<std::endl;

    // bias init
    auto b1 = ops::Variable(scope, {1, 3}, DT_FLOAT);
    auto assign_b1 = ops::Assign(scope, b1, ops::RandomNormal(scope, {1, 3}, DT_FLOAT));
    std::cout<<"simple_networl test6"<<std::endl;

    auto b2 = ops::Variable(scope, {1, 2}, DT_FLOAT);
    auto assign_b2 = ops::Assign(scope, b2, ops::RandomNormal(scope, {1, 2}, DT_FLOAT));
    std::cout<<"simple_networl test7"<<std::endl;

    auto b3 = ops::Variable(scope, {1, 1}, DT_FLOAT);
    auto assign_b3 = ops::Assign(scope, b3, ops::RandomNormal(scope, {1, 1}, DT_FLOAT));
    std::cout<<"simple_networl test8"<<std::endl;

    // layers
    auto layer_1 = ops::Tanh(scope, ops::Tanh(scope, ops::Add(scope, ops::MatMul(scope, x, w1), b1)));
    auto layer_2 = ops::Tanh(scope, ops::Add(scope, ops::MatMul(scope, layer_1, w2), b2));
    auto layer_3 = ops::Tanh(scope, ops::Add(scope, ops::MatMul(scope, layer_2, w3), b3));
    std::cout<<"simple_networl test9"<<std::endl;

    // regularization
    auto regularization = ops::AddN(
        scope, std::initializer_list<Input>{ops::L2Loss(scope, w1),ops::L2Loss(scope, w2),ops::L2Loss(scope, w3)});
    std::cout<<"simple_networl test10"<<std::endl;

    // loss calculation
    Tensor loss_const_tensor(0.01f);
    auto loss_const = ops::Const(scope, {loss_const_tensor}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    auto loss = ops::Add(scope, ops::ReduceMean(scope, ops::Square(scope, ops::Sub(scope, layer_3, y)), {0, 1}),
                         ops::Mul(scope, loss_const, regularization)); //cast

    std::cout<<"simple_networl test11"<<std::endl;
    // add the gradients operations to the graph
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, {w1, w2, w3, b1, b2, b3}, &grad_outputs));

    Tensor apply_w1_t1(0.01f);
    auto apply_v1 = ops::Const(scope, {apply_w1_t1}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    Tensor apply_w2_t2(0.01f);
    auto apply_v2 = ops::Const(scope, {apply_w2_t2}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    Tensor apply_w3_t3(0.01f);
    auto apply_v3 = ops::Const(scope, {apply_w3_t3});  //ops::Variable(scope, 0.01f, DT_FLOAT)
    Tensor apply_w4_t4(0.01f);
    auto apply_v4 = ops::Const(scope, {apply_w4_t4}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    Tensor apply_w5_t5(0.01f);
    auto apply_v5 = ops::Const(scope, {apply_w5_t5}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    Tensor apply_w6_t6(0.01f);
    auto apply_v6 = ops::Const(scope, {apply_w6_t6}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    // update the weights and bias using gradient descent
    auto apply_w1 = ops::ApplyGradientDescent(
        scope, w1, apply_w1_t1, {grad_outputs[0]}); //cast
    auto apply_w2 = ops::ApplyGradientDescent(
        scope, w2, apply_w2_t2, {grad_outputs[1]}); //cast
    auto apply_w3 = ops::ApplyGradientDescent(
        scope, w3, apply_w3_t3, {grad_outputs[2]}); //cast
    auto apply_b1 = ops::ApplyGradientDescent(
        scope, b1, apply_w4_t4, {grad_outputs[3]}); //cast
    auto apply_b2 = ops::ApplyGradientDescent(
        scope, b2, apply_w5_t5, {grad_outputs[4]}); //cast
    auto apply_b3 = ops::ApplyGradientDescent(
        scope, b3, apply_w6_t6, {grad_outputs[5]}); //cast

    ClientSession session(scope);
    std::vector<Tensor> outputs;

    // init the weights and biases by running the assigns nodes once
    TF_CHECK_OK(session.Run({assign_w1, assign_w2, assign_w3, assign_b1, assign_b2, assign_b3}, nullptr));

    // training steps 5000 100
    for (int i = 0; i < 10; ++i)
    {
        if (i % 2 == 0)
        {
            TF_CHECK_OK(session.Run({{x, x_data}, {y, y_data}}, {loss}, &outputs));
            std::cout << "Loss after " << i << " steps " << outputs[0].scalar<float>() << std::endl;
        }
        // nullptr because the output from the run is useless
        TF_CHECK_OK(session.Run({{x, x_data}, {y, y_data}}, {apply_w1, apply_w2, apply_w3, apply_b1, apply_b2, apply_b3}, nullptr));
    }

    TF_CHECK_OK(session.Run({{x, x_data}, {y, y_data}}, {loss}, &outputs));
    std::cout << "Loss after last steps " << outputs[0].scalar<float>() << std::endl;

    // prediction using the trained neural net
    TF_CHECK_OK(session.Run({{x, {data_helper.input(110000.f, Fuel::DIESEL, 7.f)}}}, {layer_3}, &outputs));
    std::cout << "DNN output: " << *outputs[0].scalar<float>().data() << std::endl;
    std::cout << "Price predicted " << data_helper.output(*outputs[0].scalar<float>().data()) << " euros" << std::endl;

    // saving the model
    //GraphDef graph_def;
    //TF_ASSERT_OK(scope.ToGraphDef(&graph_def));
}

void compute(){
    DataHelper<iRRAM::REAL> data_helper("/home/byron/Documents/workspace/NeuralNetworFloatAbnormal/simple_network/data/", "normalized_car_features.csv");
    Tensor x_data(DataTypeToEnum<iRRAM::REAL>::v(),
                  TensorShape{static_cast<int>(data_helper.x().size()) / 3, 3});
    copy_n(data_helper.x().begin(), data_helper.x().size(),
           x_data.flat<iRRAM::REAL>().data());

    Tensor y_data(DataTypeToEnum<iRRAM::REAL>::v(),
                  TensorShape{static_cast<int>(data_helper.y().size()), 1});
    copy_n(data_helper.y().begin(), data_helper.y().size(),
           y_data.flat<iRRAM::REAL>().data());

    Scope scope = Scope::NewRootScope();

    std::cout<<"simple_networl test1"<<std::endl;
    auto x = ops::Placeholder(scope, DT_REAL);
    auto y = ops::Placeholder(scope, DT_REAL);
    std::cout<<"simple_networl test2"<<std::endl;

    // weights init
    auto w1 = ops::Variable(scope, {3, 3}, DT_REAL);
    auto assign_w1 = ops::Assign(scope, w1, ops::RandomNormal(scope, {3, 3}, DT_REAL));
    std::cout<<"simple_networl test3"<<std::endl;

    auto w2 = ops::Variable(scope, {3, 2}, DT_REAL);
    auto assign_w2 = ops::Assign(scope, w2, ops::RandomNormal(scope, {3, 2}, DT_REAL));
    std::cout<<"simple_networl tes4"<<std::endl;

    auto w3 = ops::Variable(scope, {2, 1}, DT_REAL);
    auto assign_w3 = ops::Assign(scope, w3, ops::RandomNormal(scope, {2, 1}, DT_REAL));
    std::cout<<"simple_networl tes5"<<std::endl;

    // bias init
    auto b1 = ops::Variable(scope, {1, 3}, DT_REAL);
    auto assign_b1 = ops::Assign(scope, b1, ops::RandomNormal(scope, {1, 3}, DT_REAL));
    std::cout<<"simple_networl test6"<<std::endl;

    auto b2 = ops::Variable(scope, {1, 2}, DT_REAL);
    auto assign_b2 = ops::Assign(scope, b2, ops::RandomNormal(scope, {1, 2}, DT_REAL));
    std::cout<<"simple_networl test7"<<std::endl;

    auto b3 = ops::Variable(scope, {1, 1}, DT_REAL);
    auto assign_b3 = ops::Assign(scope, b3, ops::RandomNormal(scope, {1, 1}, DT_REAL));
    std::cout<<"simple_networl test8"<<std::endl;

    // layers
    auto layer_1 = ops::Tanh(scope, ops::Tanh(scope, ops::Add(scope, ops::MatMul(scope, x, w1), b1)));
    auto layer_2 = ops::Tanh(scope, ops::Add(scope, ops::MatMul(scope, layer_1, w2), b2));
    auto layer_3 = ops::Tanh(scope, ops::Add(scope, ops::MatMul(scope, layer_2, w3), b3));
    std::cout<<"simple_networl test9"<<std::endl;

    // regularization
    auto regularization = ops::AddN(
        scope, std::initializer_list<Input>{ops::L2Loss(scope, w1),ops::L2Loss(scope, w2),ops::L2Loss(scope, w3)});
    std::cout<<"simple_networl test10"<<std::endl;

    // loss calculation
    iRRAM::REAL loss_real = 0.01;
    Tensor loss_const_tensor(loss_real);
    auto loss_const = ops::Const(scope, {loss_const_tensor}); //ops::Variable(scope, 0.01f, DT_REAL)
    auto loss = ops::Add(scope, ops::ReduceMean(scope, ops::Square(scope, ops::Sub(scope, layer_3, y)), {0, 1}),
                         ops::Mul(scope, loss_const, regularization));

    //auto loss = ops::Add(scope, ops::ReduceMax(scope, ops::Square(scope, ops::Sub(scope, layer_3, y)), {0, 1}),
                         //ops::Mul(scope, loss_const, regularization));

    std::cout<<"simple_networl test11"<<std::endl;
    // add the gradients operations to the graph
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, {w1, w2, w3, b1, b2, b3}, &grad_outputs));

    iRRAM::REAL apply_w1_r1 = 0.01;
    Tensor apply_w1_t1(apply_w1_r1);
    auto apply_v1 = ops::Const(scope, {apply_w1_t1}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    
    iRRAM::REAL apply_w1_r2 = 0.01;
    Tensor apply_w2_t2(apply_w1_r2);
    auto apply_v2 = ops::Const(scope, {apply_w2_t2}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    
    iRRAM::REAL apply_w3_r3 = 0.01;
    Tensor apply_w3_t3(apply_w3_r3);
    auto apply_v3 = ops::Const(scope, {apply_w3_t3});  //ops::Variable(scope, 0.01f, DT_FLOAT)
    
    iRRAM::REAL apply_w4_r4 = 0.01;
    Tensor apply_w4_t4(apply_w4_r4);
    auto apply_v4 = ops::Const(scope, {apply_w4_t4}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    
    iRRAM::REAL apply_w5_r5 = 0.01;
    Tensor apply_w5_t5(apply_w5_r5);
    auto apply_v5 = ops::Const(scope, {apply_w5_t5}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    
    iRRAM::REAL apply_w6_r6 = 0.01;
    Tensor apply_w6_t6(apply_w6_r6);
    auto apply_v6 = ops::Const(scope, {apply_w6_t6}); //ops::Variable(scope, 0.01f, DT_FLOAT)
    // update the weights and bias using gradient descent
    auto apply_w1 = ops::ApplyGradientDescent(
        scope, w1, apply_w1_t1, {grad_outputs[0]}); //cast
    auto apply_w2 = ops::ApplyGradientDescent(
        scope, w2, apply_w2_t2, {grad_outputs[1]}); //cast
    auto apply_w3 = ops::ApplyGradientDescent(
        scope, w3, apply_w3_t3, {grad_outputs[2]}); //cast
    auto apply_b1 = ops::ApplyGradientDescent(
        scope, b1, apply_w4_t4, {grad_outputs[3]}); //cast
    auto apply_b2 = ops::ApplyGradientDescent(
        scope, b2, apply_w5_t5, {grad_outputs[4]}); //cast
    auto apply_b3 = ops::ApplyGradientDescent(
        scope, b3, apply_w6_t6, {grad_outputs[5]}); //cast

    ClientSession session(scope);
    std::vector<Tensor> outputs;

    // init the weights and biases by running the assigns nodes once
    TF_CHECK_OK(session.Run({assign_w1, assign_w2, assign_w3, assign_b1, assign_b2, assign_b3}, nullptr));

    // training steps
    for (int i = 0; i < 10; ++i){
        if (i % 2 == 0){
            TF_CHECK_OK(session.Run({{x, x_data}, {y, y_data}}, {loss}, &outputs));
            auto loss_str = iRRAM::swrite(*outputs[0].scalar<iRRAM::REAL>().data(), 80);
            std::cout << "Loss after " << i << " steps " << loss_str<< std::endl;
        }
        // nullptr because the output from the run is useless
        TF_CHECK_OK(session.Run({{x, x_data}, {y, y_data}}, {apply_w1, apply_w2, apply_w3, apply_b1, apply_b2, apply_b3}, nullptr));
    }

    TF_CHECK_OK(session.Run({{x, x_data}, {y, y_data}}, {loss}, &outputs));
    auto loss_str = iRRAM::swrite(*outputs[0].scalar<iRRAM::REAL>().data(), 80);
    std::cout << "Loss after last steps " << loss_str<< std::endl;

    // prediction using the trained neural net
    std::cout<<"simple_network: prediction using the trained neural net start"<<std::endl;
    iRRAM::REAL km = 110000.0;
    iRRAM::REAL age = 7.0;
    std::vector<iRRAM::REAL> list1 = data_helper.input1(km, Fuel::DIESEL, age);
    Tensor test_data(DataTypeToEnum<iRRAM::REAL>::v(), TensorShape{1, 3});
    copy_n(list1.begin(), list1.size(), test_data.flat<iRRAM::REAL>().data());
    std::cout<<"simple_network: prediction using the trained neural net start 1"<<std::endl;
    TF_CHECK_OK(session.Run({{x, test_data}}, {layer_3}, &outputs));
    std::cout<<"simple_network: prediction using the trained neural net start 2"<<std::endl;
    auto dnn_output = iRRAM::swrite(*outputs[0].scalar<iRRAM::REAL>().data(), 80);
    std::cout << "DNN output: " << dnn_output<< std::endl;
    auto price_output = iRRAM::swrite(data_helper.output(*outputs[0].scalar<iRRAM::REAL>().data()), 80);
    std::cout << "Price predicted " << price_output<< " euros" << std::endl;

    // saving the model
    //GraphDef graph_def;
    //TF_ASSERT_OK(scope.ToGraphDef(&graph_def));
}
