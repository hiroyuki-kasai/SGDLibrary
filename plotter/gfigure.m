function ret = gfigure(varargin)
	% GFIGURE is...
	% - A package to let figure windows appear automatically in grid style alignment
	% - A command to batch resize and organize figures in grid style
	% GFIGURE supports multiscreens. You can specify which screen to show figure windows on.
	% Automatic alignment requires some additional settings (see "Automatic Alignment" section)
    % 
	% ### Examples:
	%     GFIGURE               ->  Align all visible figures in grid style
	%     GFIGURE 1:5           ->  Align specified figures in grid style
	%     GFIGURE 400 300       ->  Resize figure to 400x300 and align (setting is remembered)
	%     GFIGURE disp 2        ->  (-d) Show on second display (setting is remembered)
	%     GFIGURE row           ->  (-r) Align row-wise ('col' or '-s' for column-wise. setting is remembered)
	%     GFIGURE nofocus       ->  (-n) Don't give focus and only align
	%     GFIGURE tail          ->  (-t) Move figures to end
	%     GFIGURE size 400 300  ->  (-s) Explicitly specify figure size
	% Composite Calls are possible:
	%     GFIGURE 400 300 row 1:5 disp 2
	%     GFIGURE 1 400 300 row           ->  Ambiguous call. GFIGURE guesses min(1,300) is the figure handle
	%     GFIGURE('disp',[3 1 2],400,300,[1 3 5 7 9])
    % 
	% ### Automatic Alignment
	% By executing the code below, you can let figure windows automatically appear in grid style without any commands:
	%     set(0,'defaultfigurecreatefcn',@(varargin) gfigure('tail','nofocus',varargin{1}));
	% By writing this code in startup.m, figrues will appear in grid style on startup of MATLAB.
	% (see http://www.mathworks.com/help/matlab/ref/startup.html for details on startup.m)
	% The above code alters the 'defaultfigurecreatefcn' property of 'figure'
	% so that the 3rd argument is always called on figure window creation (this feature is currently undocumented).

	% Author:
	% 12.12.2014, Hikaru Ikuta

    %======================================================================
    % Settings
    %======================================================================
    defaultFigsize = [400, 300];  % Default figure size
    defaultOrientation = 'row';   % default orientation ('row': left to right, 'col': up to down)
    defaultDispPriority = [2 1];  % Order of displays to show figures on
                                  % The main display is index no. 1, and the subdisplays are indices no. 2, 3, ...    
    marginLeft = 16;              % Left Margin
    marginTop = 95;               % Top Margin
    %======================================================================
    persistent figsize
    persistent orientation
    persistent dispPriority
    if isempty(figsize)
        figsize = defaultFigsize;
    end
    if isempty(orientation)
        orientation = defaultOrientation;
    end
    if isempty(dispPriority)
        dispPriority = defaultDispPriority;
    end
    %======================================================================
    defaultArgs = struct('fighandles',sort(get(0, 'Children')));
	args = parseInputs(defaultArgs,varargin);
	if isfield(args,'figsize')
		figsize = args.figsize;
	end
	if isfield(args,'orientation')
		orientation = args.orientation;
	end
	if isfield(args,'dispPriority')
	    dispPriority = args.dispPriority;
	end
    % Alignment
	if isfield(args,'fighandles')
		% Determining subdisplay priority settings
	    dispCoords = get(0,'MonitorPosition');
	    dispArray = 1:size(dispCoords,1);
	    for dp = dispPriority
	    	if dp <= max(dispArray)
	    		dispArray(find(dispArray == dp)) = [];
	    	else
	    		dispPriority(find(dispPriority == dp)) = [];
	    	end
	    end
	    dispArray = [dispPriority dispArray];
	    dispCoords = dispCoords(dispArray,:)';
	    dispIndBounds = zeros(3,length(dispArray));
	    for ind = 1:size(dispCoords,2)
	    	screenSize = dispCoords(:,ind);
	        rowMax = floor((screenSize(3) - screenSize(1)) / (marginLeft + figsize(1)));
	        colMax = floor((screenSize(4) - screenSize(2)) / (marginTop + figsize(2)));
	        dispIndBounds(:,ind) = [rowMax; colMax; rowMax*colMax];
	    end
	    mainscreenSize = get(0,'screenSize');
		for ind = 1:length(args.fighandles)
			if ~isfield(args,'nofocus')
				figure(args.fighandles(ind));
			else
				currentFigures = get(0,'Children');
				if isempty(find(currentFigures == args.fighandles(ind)))
		            warning('gfigure:warning', ['Figure ' str2double(args.fighandles(ind)) ' does not exist']);
		            continue
		        end
		    end
			if isfield(args,'tail')
				cumulativeIndex = ind + length(get(0,'Children')) - 1;
			else
				cumulativeIndex = ind;
			end
			figNum = args.fighandles(ind);			
		    for targetDisp = 1:size(dispCoords,2)
		    	if cumulativeIndex <= sum(dispIndBounds(3,1:targetDisp))
		    		break;
		    	end
		    end
		    numPos = cumulativeIndex - sum(dispIndBounds(3,1:(targetDisp-1)));
		    numPos = min(numPos, dispIndBounds(3,targetDisp));
		    % Determime display to show on
	        if strcmp(orientation,'col')
	            row = ceil(numPos / dispIndBounds(2,targetDisp)) - 1;
	            col = mod(numPos - 1, dispIndBounds(2,targetDisp)) + 1;
	        else
	            row = mod(numPos - 1, dispIndBounds(1,targetDisp));
	            col = ceil(numPos / dispIndBounds(1,targetDisp));
	        end
	        hLength = marginLeft + figsize(1);
	        vLength = marginTop + figsize(2);
	        left = dispCoords(1,targetDisp) + marginLeft + hLength * row;
	        bottom = mainscreenSize(4) - dispCoords(2,targetDisp) - vLength * col;
	        set(figNum,'Position',[left, bottom, figsize(1), figsize(2)]);
		end
	end
end
function args = parseInputs(defaultArgs,inArgs)
	args = defaultArgs;
	ind = 1;
	while ind <= length(inArgs)
		% If 3 consecutive arguments (starting from current) are numerics of length 1
		if isNumericArg(inArgs{ind}) && (length(inArgs) - ind) >= 2 && isNumericArg(inArgs{ind+1}) && isNumericArg(inArgs{ind+2}) ...
			   && isscalar(evalNumericArg(inArgs{ind})) && isscalar(evalNumericArg(inArgs{ind+1})) && isscalar(evalNumericArg(inArgs{ind+2}))
            warning('gfigure:warning', ['Ambiguous call to gfigure: parsing smallest and left/rightmost integer argument as figure handle']);
			indexIsThirdArg = (evalNumericArg(inArgs{ind}) > evalNumericArg(inArgs{ind+2})) + 0;
			args.fighandles = evalNumericArg(inArgs{ind+2*indexIsThirdArg});
			args.figsize = [evalNumericArg(inArgs{ind+1*(1-indexIsThirdArg)}), evalNumericArg(inArgs{ind+1+1*(1-indexIsThirdArg)})];
			ind = ind + 2;
		% If 2 consecutive arguments (starting from current) are numerics of length 1
		elseif isNumericArg(inArgs{ind}) && (length(inArgs) - ind) >= 1 && isNumericArg(inArgs{ind+1}) ...
			   && isscalar(evalNumericArg(inArgs{ind})) && isscalar(evalNumericArg(inArgs{ind+1}))
			args.figsize = [evalNumericArg(inArgs{ind}), evalNumericArg(inArgs{ind+1})];
			ind = ind + 1;
		% If argument is numeric
		elseif isNumericArg(inArgs{ind}) && ~isempty(evalNumericArg(inArgs{ind}))
			args.fighandles = evalNumericArg(inArgs{ind});
		% If argument is not a numeric
		elseif ~isNumericArg(inArgs{ind})
			switch lower(inArgs{ind})
				case {'row','-r'}
					args.orientation = 'row';
				case {'col','-c'}
					args.orientation = 'col';
				case {'tail','-t'}
					args.tail = true;
				case {'nofocus','-n'}
					args.nofocus = true;
				case {'disp','-d'}
					if (length(inArgs) - ind) >= 1 && isNumericArg(inArgs{ind+1})
						args.dispPriority = evalNumericArg(inArgs{ind+1});
						ind = ind + 1;
					end
				case {'size','-s'}
					if (length(inArgs) - ind) >= 2 && isNumericArg(inArgs{ind+1}) && isNumericArg(inArgs{ind+2}) ...
			           && isscalar(evalNumericArg(inArgs{ind+1})) && isscalar(evalNumericArg(inArgs{ind+2}))
						args.figsize = [evalNumericArg(inArgs{ind+1}), evalNumericArg(inArgs{ind+2})];
						ind = ind + 2;
					end
			end
		end
		ind = ind + 1;
	end
end
function ret = evalNumericArg(arg)
	if isnumeric(arg)
		ret = arg;
	elseif ~isempty(regexp(arg,'^\d+ *: *\d+$','once'))
		C = textscan(arg, '%s', 'delimiter', ':');
		ret = str2double(C{1}{1}):str2double(C{1}{2});
	elseif ~isempty(regexp(arg,'^ *\d+ *$','once'))
		ret = str2double(arg);
	else
		ret = arg;
	end
end
function ret = isNumericArg(arg)
	ret = isnumeric(evalNumericArg(arg));
end