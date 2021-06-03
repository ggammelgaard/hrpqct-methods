function start_parpool()

    if isunix  % If running on root in linux
        warning('off','parallel:cluster:FileStorageNotWritable')  % warning that i fix here
        myCluster = parcluster('local');
        warning('on','parallel:cluster:FileStorageNotWritable')
        myCluster.JobStorageLocation = '/mnt/data/joe/.matlab';
        saveProfile(myCluster);
    end

    % Get cores
    n_cores = feature('numcores');
    % Init parpool
    if isempty(gcp('nocreate'))  % if pool is not already running
        parpool('local',n_cores);
    end
end