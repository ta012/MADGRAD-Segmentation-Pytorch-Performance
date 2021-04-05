def sanity_check(model):
	for p in model.parameters():
		if not p.requires_grad:
			exit('some layer frozen')