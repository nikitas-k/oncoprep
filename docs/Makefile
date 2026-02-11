.PHONY: help html clean livehtml

help:
	@echo "  html       build HTML documentation"
	@echo "  clean      remove build artifacts"
	@echo "  livehtml   auto-rebuild on changes (requires sphinx-autobuild)"

html:
	sphinx-build -b html . _build/html

clean:
	rm -rf _build

livehtml:
	sphinx-autobuild . _build/html
