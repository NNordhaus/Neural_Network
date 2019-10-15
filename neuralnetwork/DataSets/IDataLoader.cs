using NeuralNetwork.Classes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.DataSets
{
    interface IDataLoader
    {
        IList<DataSet> GetDataSet(string imageFilePath, string labelFilePath, int count);
    }
}
