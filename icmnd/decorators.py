
def overrides(interface_class):
  """
  Checks the method definition actually overrides a method in a super class
  or interface `interface_class`.
  :param interface_class: The super class
  :return: The method itself.
  """
  def overrider(method):
    assert (method.__name__ in dir(interface_class))
    return method
  return overrider
