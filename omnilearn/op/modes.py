
import sys, os

from contextlib import nullcontext, redirect_stdout, redirect_stderr


import omnifig as fig

# class LogFile_Arg(fig.Meta_Argument, name='logfile', code='l', num_params=1):
# 	pass
#
# class Logging_Mode(fig.Run_Mode, supports=[LogFile_Arg]):
#
# 	# def __init__(self, meta, config, auto_meta_args=[]):
# 	# 	pass
#
# 	def run(self, script_info, meta, config):
#
# 		ctxt, ctxt2 = nullcontext(), nullcontext()
#
# 		if 'JOBDIR' in os.environ:
# 			proc = os.environ['PROCESS_ID']
# 			logpath = os.path.join(os.environ['JOBDIR'], f'out-{proc}.txt')
#
# 			ctxt = redirect_stderr(sys.stdout)
# 			ctxt2 = redirect_stdout(open(logpath, 'a+'))
#
# 		with ctxt, ctxt2:
# 			return super().run(script_info, meta, config)
#
# 	pass



