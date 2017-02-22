import os
import numpy

def load_bin_vec(fname):
  """
  Loads 300x1 word vecs from Google (Mikolov) word2vec
  """
  print fname
  word_vecs = {}
  with open(fname, "rb") as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = numpy.dtype('float32').itemsize * layer1_size
    print 'vocab_size, layer1_size', vocab_size, layer1_size
    count = 0
    for i, line in enumerate(xrange(vocab_size)):
      if i % 100000 == 0:
        print '.',
      word = []
      while True:
        ch = f.read(1)
        if ch == ' ':
            word = ''.join(word)
            break
        if ch != '\n':
            word.append(ch)
      count += 1
      word_vecs[word] = numpy.fromstring(f.read(binary_len), dtype='float32')
    print "done"
    print "Words found in wor2vec embeddings", count
    return word_vecs

def main():
  dir = 'data/embedding/'
  word2vec = load_bin_vec(dir + 'aquaint+wiki.txt.gz.ndim=50.bin')
  print(word2vec['hello'])
  outfile = open(dir+'aquaint.word2vec.50d.txt', 'w')
  for word in word2vec:
    outfile.write(word)
    for i in range(word2vec[word].size):
      outfile.write(' ' + str(word2vec[word][i]))
    outfile.write('\n')
  outfile.close()
if __name__ == '__main__':
  main()
