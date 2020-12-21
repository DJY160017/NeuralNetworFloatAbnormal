#include <iostream>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/c/c_api.h>

using namespace std;
using namespace tensorflow;

int main() {
    Session *session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        cout << status.ToString() << endl;
        return 1;
    }

    cout << "Session successfully created." << endl;
    cout << "Hello from Tensorflow C library version" << TF_Version() << endl;
    return 0;
}