from absl import flags
import os, sys, time, argparse

FLAGS = flags.FLAGS
FLAGS(sys.argv)

if FLAGS.loss == 'FS' and FLAGS.topic == 'ELeGANt':
	from .arch_FS import *
else:
	from .arch_base import *
if FLAGS.gan == 'WAE':
	from .arch_WAE import *