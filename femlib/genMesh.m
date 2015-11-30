function genMesh(mesh_diam, filename)
[p,e,t] = initmesh('@circleg', 'Hmax', mesh_diam);
save(filename, 'p', 'e', 't');
end
