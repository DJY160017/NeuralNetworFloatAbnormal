#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iRRAM/lib.h>

enum class Fuel {
    DIESEL,
    GAZOLINE
};

// Meta data used to normalize the data set. Useful to
// go back and forth between normalized data.
template <typename T>
class MetaData {
public:
  T mean_km;
  T std_km;
  T mean_age;
  T std_age;
  T min_price;
  T max_price;
};

template <typename T>
class DataHelper {
public:
  // Construct a data set from the given csv file path.
  DataHelper(std::string dir, std::string file_name) {
    ReadCSVFile(dir, file_name);
  }

  // getters
  std::vector<T>& x() { return x_; }
  std::vector<T>& y() { return y_; }

  // read the given csv file and complete x_ and y_
  void ReadCSVFile(std::string dir, std::string file_name);

  // normalize a human input using the data set metadata
  std::initializer_list<T> input(T km, Fuel fuel, T age);
  std::vector<T> input1(T km, Fuel fuel, T age);

  // convert a price outputted by the DNN to a human price
  T output(T price);

private:
  MetaData<T> meta_data;
  std::vector<T> x_;
  std::vector<T> y_;

   // convert one csv line to a vector of T
  std::vector<T> ReadCSVLine(std::string line);
};


template <typename T>
std::vector<T> DataHelper<T>::ReadCSVLine(std::string line){
    std::vector<T> line_data;
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')){
        line_data.push_back(stod(cell));
    }
    return line_data;
}

template <typename T>
void DataHelper<T>::ReadCSVFile(std::string dir, std::string file_name){
    std::ifstream file(file_name);
    if (!file){
        file.open(dir + file_name);
    }
    if (!file){
        std::cerr << "ERROR: No " << file_name << " next to the binary or at " << dir
                  << ", please double check the location of the CSV dataset file." << std::endl;
        std::cerr << "ERROR: The dir option must be relative to the binary location." << std::endl;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string line;
    std::vector<std::string> lines;
    while (getline(buffer, line, '\n')){
        lines.push_back(line);
    }

    // the first line contains the metadata
    std::vector<T> metadata = ReadCSVLine(lines[0]);

    meta_data.mean_km = metadata[0];
    meta_data.std_km = metadata[1];
    meta_data.mean_age = metadata[2];
    meta_data.std_age = metadata[3];
    meta_data.min_price = metadata[4];
    meta_data.max_price = metadata[5];

    // the other lines contain the features for each car
    for (int i = 2; i < lines.size(); ++i){
        std::vector<T> features = ReadCSVLine(lines[i]);
        x_.insert(x_.end(), features.begin(), features.begin() + 3);
        y_.push_back(features[3]);
    }
}

template <typename T>
std::initializer_list<T> DataHelper<T>::input(T km, Fuel fuel, T age){
    km = (km - meta_data.mean_km) / meta_data.std_km;
    age = (age - meta_data.mean_age) / meta_data.std_age;
    T f = fuel == Fuel::DIESEL ? -1.f : 1.f;
    return {km, f, age};
}

template <typename T>
std::vector<T> DataHelper<T>::input1(T km, Fuel fuel, T age){
    std::vector<T> result;
    km = (km - meta_data.mean_km) / meta_data.std_km;
    age = (age - meta_data.mean_age) / meta_data.std_age;
    T f = fuel == Fuel::DIESEL ? -1.0 : 1.0;
    result.push_back(km);
    result.push_back(f);
    result.push_back(age);
    return result;
}

template <typename T>
T DataHelper<T>::output(T price){
    return price * (meta_data.max_price - meta_data.min_price) + meta_data.min_price;
}