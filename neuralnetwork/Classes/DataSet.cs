using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Classes
{
    public class DataSet
    {
        public double[] Values { get; set; }
        public double[] Targets { get; set; }

        public DataSet(double[] values, double[] targets)
        {
            Values = values;
            Targets = targets;
        }
    }
}