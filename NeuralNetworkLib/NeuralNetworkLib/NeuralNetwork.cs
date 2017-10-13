using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;

namespace NeuralNetworkLib
{
    public class NeuralNetwork
    {
        MatrixBuilder<double> MatrixEngine = Matrix<double>.Build;
        Random RandomEngine = new Random();

        public InputLayer InputLayer { get; set; }
        public HiddenLayer HiddenLayer { get; set; }
        public OutputLayer OutputLayer { get; set; }

        public NeuralNetwork(Matrix<double> Inputs, int nbHiddenLayerNeurons, Matrix<double> Outputs)
        {
            this.InputLayer = new InputLayer(Inputs);
            this.OutputLayer = new OutputLayer(Outputs);
            this.HiddenLayer = new HiddenLayer(MatrixEngine.Dense(nbHiddenLayerNeurons, 1));
            GenerateWeight();
        }

        public void GenerateWeight()
        {
            this.InputLayer.Weights = MatrixEngine.Dense(InputLayer.Inputs.ColumnCount, this.HiddenLayer.Values.RowCount, (x, j) => RandomEngine.NextDouble());
            this.HiddenLayer.Weights = MatrixEngine.Dense(HiddenLayer.Values.RowCount, this.OutputLayer.Values.ColumnCount, (x, j) => RandomEngine.NextDouble());

        }

        public double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }



        public Matrix<double> ApplyActionMatrix(Matrix<double> matrix , Func<double,double> methode )
        {
            for (int i = 0; i < matrix.RowCount; i++)
            {
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    matrix[i, j] =  methode(matrix[i,j]);
                }
            }
            return matrix;
        }
        public void FowardPropagation()
        {
            HiddenLayer.Values = InputLayer.Inputs.Multiply(InputLayer.Weights);
            HiddenLayer.ActivateValues = ApplyActionMatrix(HiddenLayer.Values.Clone(), Sigmoid);

            OutputLayer.Values = HiddenLayer.Values.Multiply(HiddenLayer.Weights);
            OutputLayer.ActivatedValues = ApplyActionMatrix(OutputLayer.Values.Clone(), Sigmoid);


        }

    }


    public class InputLayer
    {
        public Matrix<double> Inputs { get; set; }
        public Matrix<double> Weights { get; set; }

        public InputLayer(Matrix<double> inputs)
        {

            Inputs = inputs;
        }


    }

    public class HiddenLayer
    {
        public Matrix<double> Values { get; set; }
        public Matrix<double> ActivateValues { get; set; }
        public Matrix<double> Weights { get; set; }

        public HiddenLayer(Matrix<double> values)
        {
            Values = values;
            ActivateValues = values.Clone();

        }
    }

    public class OutputLayer
    {
        public Matrix<double> Values { get; set; }
        public Matrix<double> ActivatedValues { get; set; }
        public Matrix<double> ExpectedInput { get; set; }

        public OutputLayer(Matrix<double> values)
        {
            Values = values;
            ActivatedValues = values.Clone();
            ExpectedInput = Values.Clone();
        }

    }
}
