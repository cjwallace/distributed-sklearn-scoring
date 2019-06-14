.PHONY: package-environment, package-model

package-environment:
	conda create -n distribution_env -y -q python=3.6
	cp ~/.local/lib/python3.6/site-packages/* ~/.conda/envs/distribution_env/lib/python3.6/site-packages/
	cd ~/.conda/envs
	zip -r ../../distribution_env.zip distribution_env
	cd ~

package-model:
	zip -r model.pkl model.pkl.zip
