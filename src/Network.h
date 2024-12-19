#include <Eigen/Dense>

class Ann {
    int numberOfLayers;
    int numberOfInputs;
    int numberOfOutputs;
    Layer* layers; 
    
};

class Layer {
    friend class Ann;

    private:
    Eigen::MatrixXd         weights;
    const int               numberOfNodes;                  // weight column
    const int               previousLayerNumberOfNodes;     // weight rows
    
    Layer(const int& numRows, const int& numColumns);       // random valued matrix
    Layer(const Eigen::MatrixXd& m);                        // predefined matrix

    const Eigen::MatrixXd& getMatrix () const;              // return a const reference to matrix
    const int& getNumberOfNodes () const;
    const int& getPreviousLayerNumberOfNodes () const;

};