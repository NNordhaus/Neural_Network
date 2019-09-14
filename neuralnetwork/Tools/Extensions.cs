using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Tools
{
    public static class Extensions
    {
        public static double MinX = double.MaxValue;
        public static double MaxX = double.MinValue;
        public static double Sigmoid(this double x)
        {
            MinX = Math.Min(x, MinX);
            MaxX = Math.Max(x, MaxX);
            return x < -40.0 ? 0.0 : x > 40.0 ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double ReLU(this double x)
        {
            MinX = Math.Min(x, MinX);
            MaxX = Math.Max(x, MaxX);
            return Math.Max(0.0d,Math.Min(1.0000,x));
        }

        public static double Derivative(this double x)
        {
            return x * (1 - x);
        }

        public static Random rng = new Random(DateTime.Now.Millisecond * 13);
        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static double Median(this IList<double> list)
        {
            int n = list.Count;
            list = list.OrderBy(a => a).ToList();
            int middle = n / 2;
            return list[middle];
        }

        public static bool DoesBestCandidateMatch(this IList<NeuralNetwork.Classes.Neuron> outputLayer, IList<double> targets)
        {
            int targetIndex = targets.IndexOf(targets.Max());
            var outputs = outputLayer.Select(n => n.Value).ToList();
            return outputs.IndexOf(outputs.Max()) == targetIndex;
        }

        /// <summary>
        /// Compares the contents of two arrays, down to 0.0001 accuracy
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool ContentsEqual(this double[] a, double[] b)
        {
            if(a.Length != b.Length)
            {
                return false;
            }

            for(int i = 0; i < a.Length; i++)
            {
                if(Math.Abs(a[i] - b[i]) > 0.0001)
                {
                    return false;
                }
            }

            return true;
        }
    }
}