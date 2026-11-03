#ifndef UTILS_DATA_TYPE_HPP
#define UTILS_DATA_TYPE_HPP

#include <array>
#include <vector>
#include <iomanip>
#include <cstddef>
#include <stdexcept>
#include <sstream>  
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>

typedef float VectorDimensionType;
typedef long VidType;

template<size_t dim>
using VectorPointType = std::array<VectorDimensionType, dim>;
   
struct VectorDataType {  
    VidType vid;  
    std::vector<VectorDimensionType> data;

    VectorDataType(){
    }


    VectorDataType(size_t dim, VidType _vid=0): vid(_vid) {
        data.reserve(dim);
        data.resize(dim);
        std::fill(data.begin(), data.end(), (VectorDimensionType)0);
    }

    VectorDataType(size_t dim, VidType _vid, const VectorDataType& vector_point): vid(_vid) {
        if (vector_point.Dimension() != dim) {
            throw std::invalid_argument("vector data dimension does not match");
            std::exit(EXIT_FAILURE);
        }
        data.reserve(dim);
        data.resize(dim);
        std::copy_n(vector_point.data.begin(), dim, data.begin());
    }

    VectorDataType(size_t dim, VidType _vid, const std::vector<VectorDimensionType>& arr) : vid(_vid) {
        if (arr.size() != dim) {
            throw std::invalid_argument("Data size does not match with vector data dimension");
            std::exit(EXIT_FAILURE);
        }
        data.reserve(dim);
        data.resize(dim);
        std::copy_n(arr.begin(), dim, data.begin());
    }

    void SetVid(VidType _vid) {
        vid = _vid;
    }

    void CopyVectorPoint(const VectorDataType& vector_point) {
        this->vid = vector_point.vid;
        this->data = vector_point.data;
    }

    void SetVectorPoint(const VectorDataType& vector_point) {
        const size_t dim = this->Dimension();
        if (vector_point.Dimension() != dim) {
            throw std::invalid_argument("vector data dimension does not match");
            std::exit(EXIT_FAILURE);
        }
        std::copy_n(vector_point.data.begin(), dim, data.begin());
    }

    void SetVectorPoint(const std::vector<VectorDimensionType>& arr) {
        const size_t dim = this->Dimension();
        if (arr.size() != dim) {
            throw std::invalid_argument("Data size does not match with vector data dimension");
            std::exit(EXIT_FAILURE);
        }
        std::copy_n(arr.begin(), dim, data.begin());
    }
  
    VectorDataType& operator=(const VectorDataType& other) {
        if (this != &other) {
            const size_t dim = this->Dimension();
            if (other.Dimension() != dim) {
                throw std::invalid_argument("Data size does not match with vector data dimension");
                std::exit(EXIT_FAILURE);
            }
            vid = other.vid;
            std::copy_n(other.data.begin(), dim, data.begin());
        }
        return *this;
    }
  
    bool operator==(const VectorDataType& other) const {  
        if (vid != other.vid) return false;
        const size_t dim = this->Dimension();
        if (other.Dimension() != dim) return false;

        for (size_t i=0; i<dim; ++i) {
            if (other.data[i] != data[i])
                return false;
        }

        return true;  
    }  
   
    bool operator!=(const VectorDataType& other) const {  
        return !(*this == other);  
    }

    bool operator<(const VectorDataType& rhs) const {
        return this->vid < rhs.vid;
    }
  
    VectorDimensionType& at(size_t k) {  
        if (k >= this->Dimension()) {  
            throw std::out_of_range("Index out of range");  
        }  
        return data[k];  
    }  

    const VectorDimensionType& at(size_t k) const {  
        if (k >= this->Dimension()) {  
            throw std::out_of_range("Index out of range");  
        }  
        return data[k];  
    }  
  
    VectorDimensionType& operator[](size_t k) {  
        if (k >= this->Dimension()) {  
            throw std::out_of_range("Index out of range");  
        }  
        return data[k];  
    }  

    const VectorDimensionType& operator[](size_t k) const {  
        if (k >= this->Dimension()) {  
            throw std::out_of_range("Index out of range");  
        }  
        return data[k];  
    } 

    size_t Dimension() const {
        return data.size();
    }

    std::string to_string(size_t print_size=4, size_t prec=6) const {
        const size_t dim = this->Dimension();
        std::stringstream ss;  

        ss << std::fixed << std::setprecision(prec);
        ss << "#" << vid << ": (";
        if (dim <= print_size*2) {
            for (size_t i=0; i<dim; ++i) {
                if (i > 0)
                    ss << ", " << data[i];
                else
                    ss << data[i];
            }
        } else {
            for (size_t i=0; i<print_size; ++i) {
                if (i > 0)
                    ss << ", " << data[i];
                else
                    ss << data[i];
            }
            ss << ", ...";
            for (size_t i=dim-print_size; i<dim; ++i) {
                ss << ", " << data[i];
            }
        }
        ss << ")";
        
        return ss.str();  
    } 
};

inline VectorDimensionType dotProduct(const std::vector<VectorDimensionType>& a, const std::vector<VectorDimensionType>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    VectorDimensionType result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

inline VectorDimensionType L2Distance(const std::vector<VectorDimensionType>& a, const std::vector<VectorDimensionType>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size");
    }

    VectorDimensionType result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return result;
}

inline VectorDimensionType L2Distance(const VectorDimensionType* a, const VectorDimensionType* b, int d) {
    VectorDimensionType result = 0.0;
    for (size_t i = 0; i < d; ++i) {
        result += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return result;
}

inline VectorDimensionType IPDistance(const VectorDimensionType* a, const VectorDimensionType* b, int d) {
    VectorDimensionType result = 0.0;
    for (size_t i = 0; i < d; ++i) {
        result += a[i] * b[i];
    }
    return 1- result;
}

#endif  // UTILS_DATA_TYPE_HPP
