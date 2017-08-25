import sh
import yaml

arch = sh.spack("arch")
print(arch)
