function genMesh(diam, filename)
[p,e,t] = initmesh('@circleg', 'Hmax', diam);
save(filename, 'p', 'e', 't', 'diam');
end
