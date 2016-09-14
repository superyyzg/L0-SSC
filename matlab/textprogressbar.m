function [lastprintlength] = textprogressbar(strCR,lastprintlength,c,maxIter)

%strPercentageLength = 10;   %   Length of percentage string (must be >5)
%strDotsMaximum      = 10;   %   The total number of dots in a progress bar

%% Main 

if strCR == -1,
    fprintf('%s',c);
    lastprintlength = 0;
elseif strCR == 1,
    % Progress bar  - termination   
    fprintf([c '\n']);
elseif strCR == 0,
    % Progress bar - normal progress
    c = floor(c);
    percentageOut = [num2str(c) '/' num2str(maxIter)];
    %percentageOut = [percentageOut repmat(' ',1,strPercentageLength-length(percentageOut)-1)];
    %nDots = floor(c/100*strDotsMaximum);
    %dotOut = ['[' repmat('.',1,nDots) repmat(' ',1,strDotsMaximum-nDots) ']'];
    strOut = percentageOut;
    
    % carriage return
    backward = repmat('\b',1,lastprintlength);
    fprintf([backward strOut]);
    lastprintlength = length(strOut);
else
    % Any other unexpected input
    error('Unsupported argument type');
end