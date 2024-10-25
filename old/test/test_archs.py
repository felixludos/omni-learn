
from omniplex import spaces, Spec
from omnilearn import get_builder, Blueprint





def test_feedforward():

	spec = Blueprint()
	spec.change_space_of('input', 10)

	builder_type = get_builder('mlp')

	builder = builder_type(
		hidden=[20, 30],

		blueprint=spec
	)

	print(builder)

	model = builder.build()

	pass


