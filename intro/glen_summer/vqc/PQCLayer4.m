classdef PQCLayer4 < nnet.layer.Layer
    % Custom PQC layer example.

    properties (Learnable)
        % Define layer learnable parameters.
        P1
        P2
        P3
        P4
        P5
        P6
        P7
        P8
    end

    methods
        function layer = PQCLayer4
            % Set layer name.
            layer.Name = "PQCLayer4";

            % Set layer description.
            layer.Description = "Layer containing a parameterized " + ...
                "quantum circuit (PQC)";

            % Initialize learnable parameter.
            layer.P1 = rand;
            layer.P2 = rand;
            layer.P3 = rand;
            layer.P4 = rand;
            layer.P5 = rand;
            layer.P6 = rand;
            layer.P7 = rand;
            layer.P8 = rand;
        end

        function Z = predict(layer,X)
            % Z = predict(layer,X) forwards the input data X through the
            % layer and outputs the result Z at prediction time.
             Z = computeZ(X, [layer.P1, layer.P2, layer.P3, layer.P4, ...
                 layer.P5, layer.P6, layer.P7, layer.P8]);
        end

        function [dLdX,dLdP1,dLdP2,dLdP3,dLdP4,dLdP5,dLdP6,dLdP7,dLdP8] = backward(layer,X,Z,dLdZ,memory)
            % Backpropagate the derivative of the loss
            % function through the layer.
            %
            % Inputs:
            %     layer   - Layer though which data backpropagates
            %     X       - Layer input data
            %     Z       - Layer output data
            %     dLdZ    - Derivative of loss with respect to layer
            %               output
            %     memory  - Memory value from forward function
            % Outputs:
            %     dLdX   - Derivative of loss with respect to layer input
            %     dLdP1  - Derivative of loss with respect to learnable
            %              parameter P1
            %     dLdP2  - Derivative of loss with respect to learnable
            %              parameter P2

            s = pi/8;
            ZPlus = computeZ(X, [layer.P1 + s, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8]);
            ZMinus = computeZ(X,[layer.P1 - s, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8]);
            dZdP1 = X.*(ZPlus - ZMinus);%X(1,:).*((ZPlus - ZMinus)./(2*sin(X(1,:).*s)));
            dLdP1 = sum(dLdZ.*dZdP1,"all");

            ZPlus = computeZ(X, [layer.P1, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5 + s, layer.P6, layer.P7, layer.P8]);
            ZMinus = computeZ(X,[layer.P1, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5 - s, layer.P6, layer.P7, layer.P8]);
            dZdP5 = X.*(ZPlus - ZMinus);%X(1,:).*((ZPlus - ZMinus)./(2*sin(X(1,:).*s)));
            dLdP5 = sum(dLdZ.*dZdP5,"all");

            ZPlus = computeZ(X, [layer.P1, layer.P2 + s, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8]);
            ZMinus = computeZ(X,[layer.P1, layer.P2 - s, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8]);
            dZdP2 = X.*(ZPlus - ZMinus);%X(2,:).*((ZPlus - ZMinus)./(2*sin(X(2,:).*s)));
            dLdP2 = sum(dLdZ.*dZdP2,"all");

            ZPlus = computeZ(X, [layer.P1, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6 + s, layer.P7, layer.P8]);
            ZMinus = computeZ(X,[layer.P1, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6 - s, layer.P7, layer.P8]);
            dZdP6 = X.*(ZPlus - ZMinus);%X(2,:).*((ZPlus - ZMinus)./(2*sin(X(2,:).*s)));
            dLdP6 = sum(dLdZ.*dZdP6,"all");

            ZPlus = computeZ(X, [layer.P1, layer.P2, layer.P3 + s, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8]);
            ZMinus = computeZ(X,[layer.P1, layer.P2, layer.P3 - s, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8]);
            dZdP3 = X.*(ZPlus - ZMinus);%X(3,:).*((ZPlus - ZMinus)./(2*sin(X(3,:).*s)));
            dLdP3 = sum(dLdZ.*dZdP3,"all");

            ZPlus = computeZ(X, [layer.P1, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7 + s, layer.P8]);
            ZMinus = computeZ(X,[layer.P1, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7 - s, layer.P8]);
            dZdP7 = X.*(ZPlus - ZMinus);%X(3,:).*((ZPlus - ZMinus)./(2*sin(X(3,:).*s)));
            dLdP7 = sum(dLdZ.*dZdP7,"all");

            ZPlus = computeZ(X, [layer.P1, layer.P2, layer.P3, layer.P4 + s, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8]);
            ZMinus = computeZ(X,[layer.P1, layer.P2, layer.P3, layer.P4 - s, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8]);
            dZdP4 = X.*(ZPlus - ZMinus);%X(4,:).*((ZPlus - ZMinus)./(2*sin(X(4,:).*s)));
            dLdP4 = sum(dLdZ.*dZdP4,"all");

            ZPlus = computeZ(X, [layer.P1, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8 + s]);
            ZMinus = computeZ(X,[layer.P1, layer.P2, layer.P3, layer.P4, ...
                                 layer.P5, layer.P6, layer.P7, layer.P8 - s]);
            dZdP8 = X.*(ZPlus - ZMinus);%X(4,:).*((ZPlus - ZMinus)./(2*sin(X(4,:).*s)));
            dLdP8 = sum(dLdZ.*dZdP8,"all");

            % Set the gradients with respect to x and y to zero
            % because the QNN does not use them during training.
            dLdX = zeros(size(X),"like",X);
        end
    end
end

function circ = ZZFeatureMap(X)
    t=0;
    p=0;
    l=2*X(1);
    u1_1 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
    l=2*X(2);
    u1_2 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
    l=2*X(3);
    u1_3 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
    l=2*X(4);
    u1_4 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
     
    l=2*(pi - X(1))*(pi - X(2));
    u1_5 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
    l=2*(pi - X(1))*(pi - X(3));
    u1_6 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
    l=2*(pi - X(1))*(pi - X(4));
    u1_7 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];

    l=2*(pi - X(2))*(pi - X(3));
    u1_8 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];
    l=2*(pi - X(2))*(pi - X(4));
    u1_9 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];

    l=2*(pi - X(3))*(pi - X(4));
    u1_10 = [cos(t/2), -exp(i*l)*sin(t/2);exp(i*p)*sin(t/2), exp(i*l+i*p)*cos(t/2)];

    gates = [hGate(1), hGate(2), hGate(3), hGate(4), ...
        unitaryGate(1,u1_1), unitaryGate(2, u1_2), ...
        unitaryGate(3,u1_3), unitaryGate(4, u1_4), ...
        cnotGate(1,2), unitaryGate(2,u1_5), cnotGate(1,2), ...
        cnotGate(1,3), unitaryGate(2,u1_6), cnotGate(1,3), ...
        cnotGate(1,4), unitaryGate(2,u1_7), cnotGate(1,4), ...
        cnotGate(2,3), unitaryGate(2,u1_8), cnotGate(2,3), ...
        cnotGate(2,4), unitaryGate(2,u1_9), cnotGate(2,4), ...
        cnotGate(3,4), unitaryGate(2,u1_10), cnotGate(3,4)];
    circ = quantumCircuit(gates,Name="ZZFeatureMap");
end

function circ = RealAmplitudes(P)
    gates = [ryGate(1,P(1)), ryGate(2,P(2)),  ...
                  ryGate(3,P(3)), ryGate(4,P(4)),  ...
                  cnotGate(1,2), cnotGate(1,3), cnotGate(1,4), ...
                  cnotGate(2,3), cnotGate(2,4), cnotGate(3,4), ...
                  ryGate(1,P(5)), ryGate(2,P(6)),  ...
                  ryGate(3,P(7)), ryGate(4,P(8))];
    circ = quantumCircuit(gates,Name="RealAmplitudes");
end


function Z = computeZ(X, P)
    numSamples = size(X,2);
    x1 = X(1,:);
    x2 = X(2,:);
    x3 = X(3,:);
    x4 = X(4,:);
    Z = zeros(1,numSamples,"like",X);
    for j = 1:numSamples
        circ1 = ZZFeatureMap([x1(j), x2(j), x3(j), x4(j)]);
        circ2 = RealAmplitudes(P);
        totaGates = [compositeGate(circ1,[1 2 3 4])
        compositeGate(circ2,[1 2 3 4])];
        myCircuit = quantumCircuit(totaGates);
        s = simulate(myCircuit);
        Z(j) = (probability(s,1,"0") + probability(s,2,"0") + ...
                probability(s,3,"0") + probability(s,4,"0"))/4 - ...
               (probability(s,1,"1") + probability(s,2,"1") + ...
                probability(s,3,"1") + probability(s,4,"1"))/4;
    end
end