# Applied by CPack per generator; leave archive/DMG layouts unchanged.
if(CPACK_GENERATOR STREQUAL "productbuild")
  # Productbuild's distribution needs a real component choice/pkg-ref, even
  # though all InferFlux install rules use the default Unspecified component.
  set(CPACK_COMPONENTS_ALL Unspecified)
  set(CPACK_PRODUCTBUILD_IDENTIFIER "ai.inferencial.inferflux")
  # This is a CLI package, not an application bundle under /Applications.
  set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")
endif()
